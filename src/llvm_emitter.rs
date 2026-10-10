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

use crate::ast::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write;

mod fixed;
use fixed::QFormat;

/// A literal's value, for reading `@bounds` at compile time.
fn const_f64_of(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::IntLit(v, _) => Some(*v as f64),
        Expr::FloatLit(v, _) => Some(*v),
        Expr::UnaryOp {
            op: UnaryOp::Neg,
            operand,
            ..
        } => const_f64_of(operand).map(|v| -v),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RuntimeObjectKind {
    String,
    Vector,
}

/// Payload fields occupy independent, aligned eight-byte slots. Their source
/// and LLVM types remain explicit; a slot is storage, not an i64 value.
#[derive(Clone)]
struct EnumVariantLayout {
    enum_name: String,
    fields: Vec<(String, String)>,
}

/// Symbols this module `declare`s in its own prelude, so a call to one needs no
/// extra declaration.
const PRELUDE_DECLARED: &[&str] = &[
    "exit",
    "free",
    "llvm.memset.p0.i64",
    "llvm.memmove.p0.p0.i64",
    "llvm.trap",
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
pub const LIBC_SYMBOLS: &[&str] = &["usleep"];

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
    "get_capture_failure_count",
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
    "get_voice_recording_state",
    "init_shadowplay_gui",
    "is_overlay_visible",
    "print",
    "print_int",
    "print_microphone_label",
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
    /// CLI/AOT executables return an integer status even for a source void main.
    /// Embedding/JIT callers retain the source function's original ABI.
    aot_entry_status: bool,
    /// The Q format the function being emitted returns, if it returns one.
    current_ret_q: Option<QFormat>,
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
    /// The width of one element behind a `GlobalMemory<T>` / `SharedMemory<T>`
    /// binding, for EVERY primitive `T`: the stride `buf[i]` steps by and the
    /// width it loads and stores. `mem_elem_types` is narrower on purpose - the
    /// block-pointer intrinsics and the GEMM recogniser read it, and `U16`
    /// there would read as the SIGNED `i16` operand of `vpdpwssd`. Indexing
    /// needs only the width.
    mem_storage_types: BTreeMap<String, String>,
    /// Source element types retain signedness after LLVM erases it.
    mem_ast_types: BTreeMap<String, String>,
    /// Map function names to their LLVM parameter types and return type
    functions: BTreeMap<String, (Vec<String>, String)>,
    /// Each program-defined function's parameter types AS ITS DEFINITION
    /// DECLARES THEM, i.e. through `emit_type`. A call site used to derive
    /// them a second time from `ast_type_to_string` through its own table,
    /// which knew neither `U32` nor arrays and fell back to `%<name>` - so
    /// `f(x)` with `x: U32` emitted `call i32 @f(%U32 %x)` and an array
    /// argument `call i32 @f(%[I32] ..)`, both types the module never
    /// defines, under "Compilation Successful!".
    ///
    /// `None` where the definition's type came from `emit_type`'s default
    /// rather than a lowering: the call site then keeps its old answer, a type
    /// clang refuses, instead of agreeing with the definition on a wrong one.
    fn_llvm_params: BTreeMap<String, Vec<Option<String>>>,
    /// Source return types, needed for unsigned arithmetic on call results.
    fn_ast_returns: BTreeMap<String, String>,
    /// Extra runtime symbols supplied by an embedding host such as CpuJit.
    /// AOT leaves this empty because its C runtime provides fewer aliases.
    host_runtime_symbols: HashSet<String>,
    /// Recognize exact unsigned rotate idioms before generic expression lowering.
    recognize_rotates: bool,
    /// Only the CPU JIT's pointer/i64 runtime has these header layouts.
    /// AOT's runtime uses a different ABI and leaves this disabled.
    optimize_runtime: bool,
    optimize_runtime_mutations: bool,
    optimize_runtime_copies: bool,
    optimize_helper_effects: bool,
    /// Source helpers proved to access only their scalar parameters/locals.
    scalar_runtime_helpers: HashSet<String>,
    /// Local slots whose fresh allocation, uses and lifetime are proven.
    runtime_locals: BTreeMap<String, RuntimeObjectKind>,
    runtime_vector_sizes: BTreeMap<String, u64>,
    native_string_handle_normalizer: Option<usize>,
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
    enum_variant_layouts: BTreeMap<String, EnumVariantLayout>,
    /// Track whether the current block already has a terminator
    block_terminated: bool,
    /// Accumulators declared `@ZeroDrift`: representation and whether the
    /// declared value is an integer. Integer values stay in their integer
    /// domain on reads and writes; fixed-point floats require conversion.
    zero_drift: BTreeMap<String, (crate::zero_drift::DriftRepr, bool)>,
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
    /// DWARF debug information (`-g`), when requested. `None` leaves every
    /// emitted module byte-for-byte as it was; see `crate::debug_info`.
    debug: Option<crate::debug_info::DebugInfo>,
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

/// The storage width of one element of a `GlobalMemory<T>` /
/// `SharedMemory<T>`, for every primitive `T`, signed or not; `None` for one
/// with no storage type here (`bool`, a `Q` format, `BF16`).
fn memory_storage_llvm_type(ty: &Type) -> Option<String> {
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
    match primitive_llvm_type(name)? {
        "ptr" | "i1" => None,
        t => Some(t.to_string()),
    }
}

/// The LLVM type of a primitive type name, or `None` for one `emit_type`
/// has no explicit lowering for (a `Q` format, `BF16`, `TF32`).
fn primitive_llvm_type(name: &str) -> Option<&'static str> {
    Some(match name {
        "I32" | "U32" | "u32" | "i32" => "i32",
        "I64" | "U64" | "u64" | "usize" | "isize" | "i64" => "i64",
        "U8" | "I8" => "i8",
        "U16" => "i16",
        "F16" | "f16" => "half",
        "F32" | "f32" => "float",
        "F64" | "f64" => "double",
        "bool" => "i1",
        "char" | "i8" | "u8" => "i8",
        "I16" | "u16" | "i16" => "i16",
        "String" | "Vec" | "ptr" => "ptr",
        _ => return None,
    })
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
        functions.insert(
            "str_to_i64".into(),
            (vec!["&String".to_string()], "i64".into()),
        );
        functions.insert(
            "ychar_to_ascii".into(),
            (vec!["char".to_string()], "i32".into()),
        );
        functions.insert("sqrtf".into(), (vec!["F32".to_string()], "float".into()));
        functions.insert(
            "math_sqrt".into(),
            (vec!["F32".to_string()], "float".into()),
        );
        functions.insert(
            "math_fmin".into(),
            (vec!["F32".to_string(), "F32".to_string()], "float".into()),
        );
        functions.insert(
            "math_fmax".into(),
            (vec!["F32".to_string(), "F32".to_string()], "float".into()),
        );

        // --- Standard Library namespaced methods ---
        functions.insert("Vec_new".into(), (vec!["I32".to_string()], "ptr".into()));
        functions.insert(
            "Vec_push".into(),
            (
                vec!["&mut Vec".to_string(), "&char".to_string()],
                "void".into(),
            ),
        );
        functions.insert(
            "Vec_free".into(),
            (vec!["&mut Vec".to_string()], "void".into()),
        );
        functions.insert("Vec_len".into(), (vec!["&Vec".to_string()], "i64".into()));
        functions.insert(
            "Vec_get_char".into(),
            (vec!["&Vec".to_string(), "usize".to_string()], "i8".into()),
        );

        functions.insert(
            "String_len".into(),
            (vec!["&String".to_string()], "i64".into()),
        );
        functions.insert(
            "String_clone".into(),
            (vec!["&String".to_string()], "ptr".into()),
        );
        functions.insert(
            "String_push".into(),
            (
                vec!["&mut String".to_string(), "char".to_string()],
                "void".into(),
            ),
        );
        functions.insert(
            "String_push_str".into(),
            (
                vec!["&mut String".to_string(), "&String".to_string()],
                "void".into(),
            ),
        );
        functions.insert(
            "String_eq".into(),
            (
                vec!["&String".to_string(), "&String".to_string()],
                "i1".into(),
            ),
        );
        functions.insert(
            "String_eq_cstr".into(),
            (
                vec!["&String".to_string(), "&char".to_string()],
                "i1".into(),
            ),
        );
        functions.insert(
            "String_char_at".into(),
            (
                vec!["&String".to_string(), "usize".to_string()],
                "i8".into(),
            ),
        );
        functions.insert(
            "String_free".into(),
            (vec!["&mut String".to_string()], "void".into()),
        );

        Self {
            output: String::new(),
            string_constants: Vec::new(),
            string_counter: 0,
            tmp_counter: 0,
            label_counter: 0,
            current_impl_target: None,
            aot_entry_status: false,
            current_ret_q: None,
            locals: BTreeMap::new(),
            locals_ast_type: BTreeMap::new(),
            pointee_types: BTreeMap::new(),
            mem_elem_types: BTreeMap::new(),
            mem_storage_types: BTreeMap::new(),
            mem_ast_types: BTreeMap::new(),
            functions,
            fn_llvm_params: BTreeMap::new(),
            fn_ast_returns: BTreeMap::new(),
            host_runtime_symbols: HashSet::new(),
            recognize_rotates: true,
            optimize_runtime: false,
            optimize_runtime_mutations: false,
            optimize_runtime_copies: false,
            optimize_helper_effects: false,
            scalar_runtime_helpers: HashSet::new(),
            runtime_locals: BTreeMap::new(),
            runtime_vector_sizes: BTreeMap::new(),
            native_string_handle_normalizer: None,
            structs: BTreeMap::new(),
            ast_structs: HashMap::new(),
            struct_field_attrs: HashMap::new(),
            enums: BTreeMap::new(),
            enum_variants: BTreeMap::new(),
            enum_variant_layouts: BTreeMap::new(),
            block_terminated: false,
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
            debug: None,
        }
    }

    /// Emit DWARF debug information describing `source`, the `.ysu` file
    /// being compiled (`Y prog.ysu -g`). `imported` names each item an
    /// `import` brought in (see `debug_info::item_names`) with the file it was
    /// parsed from: its line numbers are that file's.
    pub fn enable_debug_info(
        &mut self,
        source: &std::path::Path,
        imported: &[(String, std::path::PathBuf)],
        optimized: bool,
    ) {
        let mut d = crate::debug_info::DebugInfo::new(source);
        d.set_optimized(optimized);
        for (item, path) in imported {
            d.set_item_file(item, path);
        }
        self.debug = Some(d);
    }

    /// What the front end checked, proved or assumed, for the program to
    /// carry under `-g` (`ydb verify` reads it). A no-op without `-g`.
    /// Permit calls to runtime entrypoints the embedding host actually owns.
    pub fn set_aot_entry_status(&mut self, enabled: bool) {
        self.aot_entry_status = enabled;
    }

    pub fn register_host_runtime_symbols(&mut self, names: &[&str]) {
        self.host_runtime_symbols
            .extend(names.iter().map(|name| (*name).to_string()));
    }

    /// Enable exact rotate recognition; disabling it preserves generic shifts.
    pub fn set_recognize_rotates(&mut self, enabled: bool) {
        self.recognize_rotates = enabled;
    }

    /// Enable closed local String/Vec reads for the CPU JIT runtime ABI.
    /// Host runtime symbols must also be registered; AOT leaves this false.
    pub fn set_optimize_runtime(&mut self, enabled: bool) {
        self.optimize_runtime = enabled;
    }

    /// Inline guarded appends to closed local CPU JIT objects with capacity.
    pub fn set_optimize_runtime_mutations(&mut self, enabled: bool) {
        self.optimize_runtime_mutations = enabled;
    }

    /// Extend guarded appends to exact-width dynamic Vec sizes and bulk
    /// String copies. Mutation optimization must also be enabled.
    pub fn set_optimize_runtime_copies(&mut self, enabled: bool) {
        self.optimize_runtime_copies = enabled;
    }

    /// Preserve closed runtime-object proofs across proved scalar-only source
    /// helpers. Calls retain their original evaluation and lowering.
    pub fn set_optimize_helper_effects(&mut self, enabled: bool) {
        self.optimize_helper_effects = enabled;
    }

    /// Supply the CPU runtime's ptr -> ptr normalizer for typed String refs.
    /// This remains unset for AOT and is independent of query optimization.
    pub fn set_native_string_handle_normalizer(&mut self, address: usize) {
        self.native_string_handle_normalizer = (address != 0).then_some(address);
    }

    pub fn set_guarantees(&mut self, g: crate::guarantees::Guarantees) {
        if let Some(d) = self.debug.as_mut() {
            d.set_guarantees(g);
        }
    }

    /// Attribute what follows to `span`; returns the position to restore.
    fn dbg_enter(&mut self, span: &Span) -> Option<(usize, usize)> {
        let outer = self.debug.as_ref()?.current();
        self.dbg_move(span.line, span.col);
        outer
    }

    /// Back to the position `dbg_enter` returned, so code a compound
    /// statement emits after its body is attributed to the statement itself.
    /// The SCOPE is not restored here: a `let` opens a scope that lasts to the
    /// end of its block, past the statement itself.
    fn dbg_leave(&mut self, outer: Option<(usize, usize)>) {
        if let Some((line, col)) = outer {
            self.dbg_move(line, col);
        }
    }

    /// Attribute what follows to `(line, col)` in the current scope, writing
    /// a marker if that changes anything.
    fn dbg_move(&mut self, line: usize, col: usize) {
        let Some(d) = self.debug.as_mut() else { return };
        if d.move_to(line, col) {
            let scope = d.scope_code();
            writeln!(
                &mut self.output,
                "{}{} {} {}",
                crate::debug_info::LOC_MARKER,
                line,
                col,
                scope
            )
            .unwrap();
        }
    }

    /// Re-attribute what follows to the current position, after the scope
    /// changed under it.
    fn dbg_refresh(&mut self) {
        if let Some((line, col)) = self.debug.as_ref().and_then(|d| d.current()) {
            self.dbg_move(line, col);
        }
    }

    /// Open a lexical scope at `span` - a `{ }` block or a `for` loop - for
    /// what follows. Returns what [`Self::dbg_scope_leave`] restores.
    fn dbg_scope_enter(&mut self, span: &Span) -> Option<Option<usize>> {
        let outer = self.debug.as_mut()?.enter_scope(span.line, span.col);
        self.dbg_refresh();
        Some(outer)
    }

    /// Close the scope [`Self::dbg_scope_enter`] opened, and every scope a
    /// `let` opened inside it.
    fn dbg_scope_leave(&mut self, outer: Option<Option<usize>>) {
        if let (Some(d), Some(o)) = (self.debug.as_mut(), outer) {
            d.leave_scope(o);
            self.dbg_refresh();
        }
    }

    /// The binding `name` exists from here on. A `let`'s binding gets a scope
    /// of its own running to the end of the enclosing block - so the debugger
    /// does not show it before the `let` has run - and a loop variable the
    /// loop's scope, already open.
    fn dbg_bind(&mut self, name: &str, span: &Span, own_scope: bool) {
        let Some(d) = self.debug.as_mut() else { return };
        if own_scope {
            d.enter_scope(span.line, span.col);
        }
        d.bind(name);
        self.dbg_refresh();
    }

    /// A nested `{ }` block, in a lexical scope of its own under `-g`.
    fn emit_scoped_block(&mut self, block: &Block, ret_type: &str) {
        let scope = self.dbg_scope_enter(&block.span);
        self.emit_block_body(block, ret_type);
        self.dbg_scope_leave(scope);
    }

    /// The slot `%name` just allocated holds a Y variable. `declared` is its
    /// type as written, `storage` the LLVM type of the slot, and `init` the
    /// initialiser of an unannotated `let`.
    fn dbg_var(
        &mut self,
        name: &str,
        span: &Span,
        declared: Option<&Type>,
        init: Option<&Expr>,
        storage: &str,
        is_param: bool,
    ) {
        if self.debug.is_none() {
            return;
        }
        // `infer_ast_type` answers a call with the callee's LLVM return type,
        // so a `let s = String_new(..)` is inferred as `ptr`; the runtime's
        // string constructors are known to return a string handle.
        let inferred = match init {
            Some(Expr::Call { func, .. })
                if crate::debug_info::returns_string(&self.emit_call_target(func)) =>
            {
                Some("String".to_string())
            }
            _ => self.locals_ast_type.get(name).cloned(),
        };
        let Some(d) = self.debug.as_mut() else { return };
        let ty = d.slot_type(declared, inferred.as_deref(), storage);
        if let Some(i) = d.declare(name, span.line, span.col, is_param, ty) {
            writeln!(&mut self.output, "{}{}", crate::debug_info::VAR_MARKER, i).unwrap();
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

    fn emit_load_with_attrs(
        &mut self,
        ptr: &str,
        ty: &str,
        attrs: Option<Vec<FieldAttrKind>>,
    ) -> String {
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
                tmp,
                ty,
                ptr,
                if is_volatile { " volatile" } else { "" },
                atomic_ordering,
                align_str
            )
            .unwrap();
        } else {
            // `@gpu_uncached` is `volatile` and nothing else. It also used to
            // carry `!nontemporal`, which on a scalar x86 load compiles to the
            // same `mov` (there is no scalar non-temporal load) - and on the
            // STORE side became `movnti`, which breaks the ordering the
            // attribute exists for. See `emit_store_with_attrs`.
            let volatile_str = if is_volatile { " volatile" } else { "" };
            writeln!(
                &mut self.output,
                "  {} = load{} {}, ptr {}{}",
                tmp, volatile_str, ty, ptr, align_str
            )
            .unwrap();
        }
        tmp
    }

    fn emit_store(&mut self, val: &str, ptr: &str, ty: &str) {
        self.emit_store_with_attrs(val, ptr, ty, None)
    }

    fn emit_store_with_attrs(
        &mut self,
        val: &str,
        ptr: &str,
        ty: &str,
        attrs: Option<Vec<FieldAttrKind>>,
    ) {
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
                ty,
                val,
                ptr,
                if is_volatile { " volatile" } else { "" },
                atomic_ordering,
                align_str
            )
            .unwrap();
        } else {
            // `@gpu_uncached` is `volatile` and nothing else. The store used to
            // carry `!nontemporal` as well, which x86-64 lowers to `movnti`: a
            // non-temporal store, and the one kind of ordinary store the x86
            // memory model lets become visible BEFORE an earlier store. So
            // `c.data = v; c.status = 1;` could publish the flag ahead of the
            // data it guards - the status-flag pattern the attribute is
            // documented for. A plain volatile store is ordered after the
            // data store, and on the load side `!nontemporal` compiled to the
            // same `mov` as without it.
            let volatile_str = if is_volatile { " volatile" } else { "" };
            writeln!(
                &mut self.output,
                "  store{} {} {}, ptr {}{}",
                volatile_str, ty, val, ptr, align_str
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
    fn drift_running_sum<'e>(target: &Expr, value: &'e Expr) -> Option<(BinaryOp, &'e Expr)> {
        crate::zero_drift::running_sum(target, value)
    }

    fn emit_to_fixed(&mut self, val: &str, repr: crate::zero_drift::DriftRepr) -> String {
        let ity = repr.llvm_type();
        if repr.frac_bits() == 0 {
            let out = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {} = fptosi double {} to {}",
                out, val, ity
            )
            .unwrap();
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
        writeln!(
            &mut self.output,
            "  {} = fcmp olt double {}, 0.0",
            is_neg, scaled
        )
        .unwrap();
        let bias = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {} = select i1 {}, double -5.000000e-01, double 5.000000e-01",
            bias, is_neg
        )
        .unwrap();
        let rounded = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {} = fadd double {}, {}",
            rounded, scaled, bias
        )
        .unwrap();
        let out = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {} = fptosi double {} to {}",
            out, rounded, ity
        )
        .unwrap();
        out
    }

    /// Converts a value out of `repr`'s integer domain back to `double`.
    fn emit_from_fixed(&mut self, val: &str, repr: crate::zero_drift::DriftRepr) -> String {
        let ity = repr.llvm_type();
        let as_f = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {} = sitofp {} {} to double",
            as_f, ity, val
        )
        .unwrap();
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

    /// Converts one term to the accumulator's storage domain. Integer terms
    /// must never pass through `double`: values above 2^53 would be rounded
    /// before the supposedly exact addition even starts.
    fn emit_drift_term(
        &mut self,
        expr: &Expr,
        repr: crate::zero_drift::DriftRepr,
        integer_domain: bool,
    ) -> String {
        if integer_domain {
            let value = self.emit_expr(expr, None, None);
            let ty = self.infer_type(expr);
            return self.emit_coerce_from(
                &value,
                &ty,
                repr.llvm_type(),
                self.expr_is_unsigned(expr),
            );
        }
        let value = self.emit_expr_as_double(expr);
        self.emit_to_fixed(&value, repr)
    }

    /// Emits an expression in the conversion domain of a fixed-point float.
    fn emit_expr_as_double(&mut self, expr: &Expr) -> String {
        let v = self.emit_expr(expr, None, None);
        let t = self.infer_type(expr);
        self.emit_coerce(&v, &t, "double")
    }

    /// Recover the signedness LLVM's integer types erase. Comparisons produce
    /// booleans; their operands' signedness must not propagate to the result.
    fn expr_is_unsigned(&self, e: &Expr) -> bool {
        match e {
            Expr::BinaryOp {
                left, op, right, ..
            } => {
                !matches!(
                    op,
                    BinaryOp::Eq
                        | BinaryOp::NotEq
                        | BinaryOp::Lt
                        | BinaryOp::Gt
                        | BinaryOp::Le
                        | BinaryOp::Ge
                        | BinaryOp::And
                        | BinaryOp::Or
                ) && self.binary_is_unsigned(op, left, right)
            }
            Expr::UnaryOp {
                op: UnaryOp::Neg,
                operand,
                ..
            } => self.expr_is_unsigned(operand),
            _ => matches!(
                self.infer_ast_type(e).as_str(),
                "U8" | "U16" | "U32" | "U64" | "u8" | "u16" | "u32" | "u64" | "usize"
            ),
        }
    }

    /// Mixed integers use the wider width. At equal widths unsigned wins;
    /// a wider signed type can represent every value of the unsigned type.
    /// A shift's interpretation depends only on the value being shifted.
    fn binary_is_unsigned(&self, op: &BinaryOp, left: &Expr, right: &Expr) -> bool {
        let l_unsigned = self.expr_is_unsigned(left);
        if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
            return l_unsigned;
        }
        let l_ty = self.infer_type(left);
        let r_ty = self.infer_type(right);
        if [l_ty.as_str(), r_ty.as_str()]
            .iter()
            .any(|t| matches!(*t, "float" | "double" | "half"))
        {
            return false;
        }
        let r_unsigned = self.expr_is_unsigned(right);
        match (l_unsigned, r_unsigned) {
            (true, true) => true,
            (true, false) => Self::int_bits(&l_ty) >= Self::int_bits(&r_ty),
            (false, true) => Self::int_bits(&r_ty) >= Self::int_bits(&l_ty),
            (false, false) => false,
        }
    }

    fn common_operand_type(left: &str, right: &str) -> String {
        // Boolean arithmetic uses 0/1 at a real integer width.
        let left = if left == "i1" { "i32" } else { left };
        let right = if right == "i1" { "i32" } else { right };
        let float_bits = |ty| match ty {
            "double" => 64,
            "float" => 32,
            "half" => 16,
            _ => 0,
        };
        let l_float = float_bits(left);
        let r_float = float_bits(right);
        if l_float != 0 || r_float != 0 {
            if l_float >= r_float {
                left
            } else {
                right
            }
        } else if Self::int_bits(left) >= Self::int_bits(right) {
            left
        } else {
            right
        }
        .to_string()
    }

    /// Binary expressions and compound assignments must use the same operand
    /// widths and independently extend each source with its own signedness.
    fn promote_binary_values(
        &mut self,
        op: &BinaryOp,
        left: &Expr,
        right: &Expr,
        l_value: &str,
        r_value: &str,
    ) -> (String, String, String) {
        let l_ty = self.infer_type(left);
        let r_ty = self.infer_type(right);
        let common = if matches!(op, BinaryOp::And | BinaryOp::Or) {
            "i1".to_string()
        } else {
            Self::common_operand_type(&l_ty, &r_ty)
        };
        let l_unsigned = self.expr_is_unsigned(left);
        let r_unsigned = self.expr_is_unsigned(right);
        let l_value = self.emit_coerce_from(l_value, &l_ty, &common, l_unsigned);
        let r_value = self.emit_coerce_from(r_value, &r_ty, &common, r_unsigned);
        (l_value, r_value, common)
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
            // Unsigned high-bit values must stay positive when converted.
            writeln!(
                &mut self.output,
                "  {} = {} {} {} to {}",
                tmp,
                if src_unsigned { "uitofp" } else { "sitofp" },
                src_ty,
                val,
                dst_ty
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

    fn wln(&mut self, s: &str) {
        writeln!(&mut self.output, "{}", s).unwrap();
    }

    // ── Type Mapping ────────────────────────────────────────

    fn enum_value_type(&self, layout: &EnumVariantLayout) -> String {
        if self.enums.get(&layout.enum_name) == Some(&true) {
            format!("%{}", layout.enum_name)
        } else {
            "i32".into()
        }
    }

    fn enum_payload_field(&self, expr: &Expr) -> Result<Option<(Expr, EnumVariantLayout, usize)>, String> {
        let Expr::MemberAccess { base, member, .. } = expr else { return Ok(None) };
        let Expr::MemberAccess { base: data, member: variant, .. } = &**base else { return Ok(None) };
        let Expr::MemberAccess { base: owner, member: data_name, .. } = &**data else { return Ok(None) };
        if data_name != "data" { return Ok(None); }
        let owner_ty = self.infer_struct_type(owner);
        let enum_name = owner_ty.trim_start_matches('%');
        if !self.enums.contains_key(enum_name) { return Ok(None); }
        let key = format!("{}_{}", enum_name, variant);
        let layout = self.enum_variant_layouts.get(&key).ok_or_else(|| format!(
            "[LLVM host backend] enum `{enum_name}` has no variant `{variant}`"))?;
        let index = member.strip_prefix('_').and_then(|s| s.parse::<usize>().ok())
            .filter(|index| *index < layout.fields.len()).ok_or_else(|| format!(
                "[LLVM host backend] enum payload `{enum_name}::{variant}` has no field `{member}`"))?;
        Ok(Some(((**owner).clone(), layout.clone(), index)))
    }

    fn emit_enum_constructor(&mut self, name: &str, args: &[Expr]) -> Option<String> {
        let layout = self.enum_variant_layouts.get(name)?.clone();
        let tag = self.enum_variants[name];
        let ty = self.enum_value_type(&layout);
        if args.len() != layout.fields.len() {
            self.emit_errors.push(format!("[LLVM host backend] enum constructor `{name}` expects {} argument(s), got {}",
                layout.fields.len(), args.len()));
            return Some(if ty == "i32" { "0".into() } else { "zeroinitializer".into() });
        }
        if ty == "i32" { return Some(tag.to_string()); }
        let storage = self.fresh_tmp();
        writeln!(&mut self.output, "  {storage} = alloca {ty}, align 8\n  store {ty} zeroinitializer, ptr {storage}, align 8").unwrap();
        let tag_pointer = self.fresh_tmp();
        writeln!(&mut self.output, "  {tag_pointer} = getelementptr {ty}, ptr {storage}, i32 0, i32 0\n  store i32 {tag}, ptr {tag_pointer}").unwrap();
        for (index, (arg, (field_ty, _))) in args.iter().zip(&layout.fields).enumerate() {
            let value = self.emit_expr(arg, None, Some(field_ty.clone()));
            let source_ty = self.infer_type(arg);
            let value = self.emit_coerce_from(&value, &source_ty, field_ty, self.expr_is_unsigned(arg));
            let pointer = self.fresh_tmp();
            writeln!(&mut self.output, "  {pointer} = getelementptr {ty}, ptr {storage}, i32 0, i32 1, i32 {index}\n  store {field_ty} {value}, ptr {pointer}, align 8").unwrap();
        }
        Some(self.emit_load(&storage, &ty))
    }

    fn emit_type(&mut self, ty: &Type) -> String {
        let res: String = match ty {
            // `U64`/`u64` were absent from this table and fell to the
            // `"i32"` default, so `let x: U64 = ...` allocated an **i32**. The
            // PTX backend takes these types seriously (gotcha #7); this one
            // silently halved their width. A `Q` format outside `@ZeroDrift`
            // fell to the same default and was computed as an integer (`let
            // x: Q16.16 = 1.5` stored `fptosi 1.5` = 1); it is scaled integer
            // storage now, and `fixed.rs` keeps its arithmetic in that domain.
            Type::Primitive(name, _) | Type::Ident(name, _)
                if primitive_llvm_type(name).is_none() && QFormat::parse(name).is_some() =>
            {
                self.q_storage(QFormat::parse(name).unwrap())
            }
            Type::Primitive(name, _) => primitive_llvm_type(name).unwrap_or("i32").into(),
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

    /// The parameter types a definition's signature declares, for its call
    /// sites. The definition reports any type it cannot lower when it is
    /// emitted, so the errors `emit_type` pushes here would only print every
    /// such message twice.
    fn definition_param_types(&mut self, params: &[Param]) -> Vec<Option<String>> {
        let before = self.emit_errors.len();
        let tys = params
            .iter()
            .map(|p| match &p.ty {
                Type::Primitive(n, _)
                    if primitive_llvm_type(n).is_none() && QFormat::parse(n).is_none() => None,
                t => Some(self.emit_type(t)),
            })
            .collect();
        self.emit_errors.truncate(before);
        tys
    }

    /// The storage type of a local array, or of the copy an array parameter
    /// is given on entry: `[N x T]` for a literal length and a scalar element.
    ///
    /// **A local array used to have no storage at all.** `emit_type` answers
    /// `ptr` for every array - right for a parameter, which arrives as a
    /// pointer - and the `let` allocated exactly that: one pointer slot,
    /// which `= {}` then zeroed. Every element access loaded that null
    /// pointer and indexed from it, in 8-byte `i64` slots whatever the
    /// element type. Measured, each under "Compilation Successful!" and each
    /// running: `v[0] = 4; v[2] = 6; return v[0] + v[2];` exited 0 (clang
    /// deleted the stores through null as undefined behaviour); reading a
    /// zero-initialised element segfaulted; `let w = v;` segfaulted; an `F32`
    /// array stored `fptosi 1.5` = 1.
    ///
    /// Anything this cannot hold is refused by name rather than given a
    /// representation the element accesses do not share.
    fn local_array_type(&mut self, ty: &Type, what: &str, span: &Span) -> Option<String> {
        let (element, size) = match ty {
            Type::Array { element, size, .. } => (element, size),
            _ => return None,
        };
        // A whitelist on the AST, not a test of what `emit_type` answered: its
        // primitive arm ends in `_ => "i32"`, so a `Q16.16` element would have
        // looked like a scalar here and been stored in four integer bytes.
        let scalar = matches!(
            &**element,
            Type::Primitive(n, _) if matches!(
                n.as_str(),
                "I8" | "I16" | "I32" | "I64" | "U8" | "U16" | "U32" | "U64"
                    | "F16" | "F32" | "F64" | "bool" | "char"
            )
        );
        let elem = self.emit_type(element);
        let reason = match (&**size, scalar) {
            (Expr::IntLit(n, _), true) if *n > 0 => return Some(format!("[{} x {}]", n, elem)),
            (Expr::IntLit(n, _), true) => format!("its length is {}", n),
            (_, true) => "its length is not an integer literal".to_string(),
            (_, false) => format!(
                "its elements are `{}` - this backend stores arrays of scalars only",
                ast_type_to_string(element)
            ),
        };
        self.emit_errors.push(format!(
            "Line {}: [LLVM host backend] {} of type `{}` cannot be given storage: {}.",
            span.line,
            what,
            ast_type_to_string(ty),
            reason
        ));
        None
    }

    // ── Entry Point ─────────────────────────────────────────

    pub fn emit_program(
        &mut self,
        prog: &Program,
        profile: &crate::sentinel::HardwareProfile,
    ) -> String {
        // `@cache_policy` is refused here, not lowered. Its four policies are
        // NVIDIA L2 eviction priorities and x86 has no instruction that sets
        // a cache line's eviction priority. This backend used to emit a
        // mapping, and on x86-64 at clang -O2 it did nothing the directive
        // names: `L2_EVICT_FIRST`'s `!nontemporal` on a scalar load compiled
        // to the same `mov` as no policy, `L2_PERSIST`'s `llvm.prefetch` came
        // out as a `prefetcht0` scheduled AFTER the load of the same address,
        // and `L2_EVICT_LAST` and `L2_STREAM` were dropped - all four exit 0.
        for site in crate::ast::cache_policy_sites(prog) {
            self.emit_errors
                .push(crate::ast::cache_policy_refusal("LLVM backend", &site));
        }

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
        self.functions
            .insert("Vec_new".into(), (vec!["I32".to_string()], "ptr".into()));
        self.functions.insert(
            "Vec_push".into(),
            (
                vec!["&mut Vec".to_string(), "&char".to_string()],
                "void".into(),
            ),
        );
        self.functions.insert(
            "Vec_free".into(),
            (vec!["&mut Vec".to_string()], "void".into()),
        );
        self.functions
            .insert("Vec_len".into(), (vec!["&Vec".to_string()], "i64".into()));
        self.functions.insert(
            "Vec_get_char".into(),
            (vec!["&Vec".to_string(), "usize".to_string()], "i8".into()),
        );

        self.functions.insert(
            "String_len".into(),
            (vec!["&String".to_string()], "i64".into()),
        );
        self.functions.insert(
            "String_clone".into(),
            (vec!["&String".to_string()], "ptr".into()),
        );
        self.functions.insert(
            "String_push".into(),
            (
                vec!["&mut String".to_string(), "char".to_string()],
                "void".into(),
            ),
        );
        self.functions.insert(
            "String_push_str".into(),
            (
                vec!["&mut String".to_string(), "&String".to_string()],
                "void".into(),
            ),
        );
        self.functions.insert(
            "String_eq".into(),
            (
                vec!["&String".to_string(), "&String".to_string()],
                "i1".into(),
            ),
        );
        self.functions.insert(
            "String_eq_cstr".into(),
            (
                vec!["&String".to_string(), "&char".to_string()],
                "i1".into(),
            ),
        );
        self.functions.insert(
            "String_char_at".into(),
            (
                vec!["&String".to_string(), "usize".to_string()],
                "i8".into(),
            ),
        );
        self.functions.insert(
            "String_free".into(),
            (vec!["&mut String".to_string()], "void".into()),
        );

        self.functions.insert(
            "File_read_to_string".into(),
            (vec!["&String".to_string()], "ptr".into()),
        );
        self.functions.insert(
            "File_write".into(),
            (
                vec!["&String".to_string(), "&String".to_string()],
                "void".into(),
            ),
        );

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
            if let Item::Enum(e) = item {
                if !e.generic_params.is_empty() {
                    self.emit_errors.push(format!("[LLVM host backend] generic enum `{}` needs monomorphization and has no host payload layout", e.name));
                }
                for variant in &e.variants {
                    let mut fields = Vec::new();
                    for field in variant.fields.iter().flatten() {
                        let llvm_ty = match field {
                            Type::Primitive(name, _) | Type::Ident(name, _) => {
                                primitive_llvm_type(name).map(str::to_string).or_else(|| {
                                    (self.enums.get(name) == Some(&false)).then(|| "i32".into())
                                })
                            }
                            Type::Reference { .. } => Some("ptr".into()),
                            _ => None,
                        };
                        if llvm_ty.is_none() {
                            self.emit_errors.push(format!("[LLVM host backend] enum payload `{}::{}` type `{}` has no supported scalar layout; aggregate, generic and fixed-point payloads are unsupported",
                                e.name, variant.name, ast_type_to_string(field)));
                        }
                        fields.push((llvm_ty.unwrap_or_else(|| "i32".into()), ast_type_to_string(field)));
                    }
                    if fields.len() > 8 {
                        self.emit_errors.push(format!("[LLVM host backend] enum payload `{}::{}` has {} fields; this backend supports at most 8 scalar payload fields",
                            e.name, variant.name, fields.len()));
                    }
                    let key = format!("{}_{}", e.name, variant.name);
                    if self.enum_variant_layouts.insert(key.clone(), EnumVariantLayout {
                        enum_name: e.name.clone(), fields,
                    }).is_some() {
                        self.emit_errors.push(format!("[LLVM host backend] enum constructor name `{key}` is ambiguous"));
                    }
                }
            }
        }
        for item in &prog.items {
            match item {
                Item::Struct(s) => {
                    let mut fields = Vec::new();
                    let mut ast_fields = Vec::new();
                    let mut field_attrs = HashMap::new();
                    for f in &s.fields {
                        // An array of a Q format has no fixed-point lowering
                        // (a local one is refused by `local_array_type`), and
                        // its literal would be stored unscaled.
                        if let Type::Array { element, .. } = &f.ty {
                            if let Some(fmt) = QFormat::parse(&ast_type_to_string(element)) {
                                self.emit_errors.push(format!(
                                    "[LLVM host backend] struct field `{}.{}` is an array of {}; \
                                     arrays of a Q format have no fixed-point lowering on this backend",
                                    s.name, f.name, fmt.name()
                                ));
                            }
                        }
                        fields.push((f.name.clone(), self.emit_field_type(&f.ty)));
                        ast_fields.push((f.name.clone(), ast_type_to_string(&f.ty)));
                        let attrs: Vec<FieldAttrKind> =
                            f.attrs.iter().map(|attr| attr.kind.clone()).collect();
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
                        .unwrap_or_else(|| if self.aot_entry_status && f.name == "main" {
                            "i32".into()
                        } else {
                            "void".into()
                        });
                    let param_tys: Vec<String> =
                        f.params.iter().map(|p| ast_type_to_string(&p.ty)).collect();
                    self.functions.insert(f.name.clone(), (param_tys, ret_ty));
                    if let Some(ret) = &f.ret_ty {
                        self.fn_ast_returns
                            .insert(f.name.clone(), ast_type_to_string(ret));
                    }
                    let llvm = self.definition_param_types(&f.params);
                    self.fn_llvm_params.insert(f.name.clone(), llvm);
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
                        let name = format!("{}_{}", imp.target_type, m.name);
                        self.functions.insert(name.clone(), (param_tys, ret_ty));
                        if let Some(ret) = &m.ret_ty {
                            self.fn_ast_returns
                                .insert(name.clone(), ast_type_to_string(ret));
                        }
                        let llvm = self.definition_param_types(&m.params);
                        self.fn_llvm_params.insert(name, llvm);
                    }
                }
                Item::Kernel(k) => {
                    let param_tys: Vec<String> =
                        k.params.iter().map(|p| ast_type_to_string(&p.ty)).collect();
                    self.functions
                        .insert(k.name.clone(), (param_tys, "void".into()));
                    let llvm = self.definition_param_types(&k.params);
                    self.fn_llvm_params.insert(k.name.clone(), llvm);
                }
                _ => {}
            }
        }

        self.scalar_runtime_helpers = self.prove_scalar_runtime_helpers(prog);

        if let Some(d) = &mut self.debug {
            d.register_types(prog, &self.structs);
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
        if !self.fn_llvm_params.contains_key("ystr_push") {
            self.wln("declare void @ystr_push(ptr, i8)");
        }
        if !self.fn_llvm_params.contains_key("ystr_push_str") {
            self.wln("declare void @ystr_push_str(ptr, ptr)");
        }
        self.wln("declare i1 @ystr_eq_cstr(ptr, ptr)");
        self.wln("declare i64 @ystr_len(ptr)");
        self.wln("declare i8 @ystr_char_at(ptr, i64)");
        self.wln("declare ptr @ystr_clone(ptr)");
        self.wln("declare ptr @yvec_new(i64)");
        if !self.fn_llvm_params.contains_key("yvec_push") {
            self.wln("declare void @yvec_push(ptr, ptr)");
        }
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
        // No longer called: its one caller was the `L2_PERSIST` mapping,
        // and `@cache_policy` is refused now. The declaration stays so the
        // prelude, and so every module this backend emits, is unchanged.
        self.wln("declare void @llvm.prefetch.p0(ptr nocapture readonly, i32, i32, i32)");
        self.wln("declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)");
        if self
            .called_functions
            .iter()
            .any(|name| name == "llvm.memmove.p0.p0.i64")
        {
            self.wln("declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1 immarg)");
        }
        if self.called_functions.iter().any(|name| name == "llvm.trap") {
            self.wln("declare void @llvm.trap() cold noreturn nounwind");
        }
        self.wln("declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)");
        self.wln("declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)");
        for ty in ["i8", "i16", "i32", "i64"] {
            for intrinsic in ["fshl", "fshr"] {
                self.wln(&format!(
                    "declare {ty} @llvm.{intrinsic}.{ty}({ty}, {ty}, {ty})"
                ));
            }
        }
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
            .chain(self.host_runtime_symbols.iter().map(String::as_str))
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
                        writeln!(&mut extern_decls, "declare {} @{}(...)", ret_ty, fname).unwrap();
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

        // Metadata node `!0`, referenced by `!uniform_branch` on loop branches.
        // It was also the operand of `!nontemporal`, which this backend no
        // longer emits (see `emit_store_with_attrs`). Kept in every module so
        // the prelude and the emitted text of every other program are unchanged.
        self.wln("!0 = !{i32 1}");

        if let Some(d) = self.debug.take() {
            self.output = d.finish(&self.output);
        }

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

    /// Local facts belong to one function (including a kernel). Leaving any
    /// of these maps populated makes an unrelated binding with the same name
    /// inherit its predecessor's directive, signedness or pointer element.
    fn reset_function_state(&mut self) {
        self.current_ret_q = None;
        self.tmp_counter = 0;
        self.label_counter = 0;
        self.locals.clear();
        self.locals_ast_type.clear();
        self.pointee_types.clear();
        self.mem_elem_types.clear();
        self.mem_storage_types.clear();
        self.mem_ast_types.clear();
        self.zero_drift.clear();
        self.loop_exit_stack.clear();
        self.block_terminated = false;
        self.current_load_hint = None;
        self.runtime_locals.clear();
        self.runtime_vector_sizes.clear();
    }

    fn runtime_arg_local(expr: &Expr) -> Option<&str> {
        match expr {
            Expr::Ident(name, _) => Some(name),
            Expr::UnaryOp {
                op: UnaryOp::Ref { .. },
                operand,
                ..
            } => match &**operand {
                Expr::Ident(name, _) => Some(name),
                _ => None,
            },
            _ => None,
        }
    }

    fn host_runtime_call(&self, name: &str) -> bool {
        self.host_runtime_symbols.contains(name) && !self.fn_llvm_params.contains_key(name)
    }

    fn scalar_helper_type(&self, ty: &Type) -> bool {
        let name = match ty {
            Type::Primitive(name, _) | Type::Ident(name, _) => name,
            _ => return false,
        };
        !self.structs.contains_key(name)
            && !self.enums.contains_key(name)
            && primitive_llvm_type(name).is_some_and(|ty| ty != "ptr")
    }

    fn scalar_helper_expr(
        expr: &Expr,
        locals: &HashSet<String>,
        proved: &HashSet<String>,
        arities: &BTreeMap<String, usize>,
    ) -> bool {
        match expr {
            Expr::IntLit(..) | Expr::FloatLit(..) | Expr::CharLit(..) | Expr::BoolLit(..) => true,
            Expr::Ident(name, _) => locals.contains(name),
            Expr::BinaryOp { left, right, .. } => {
                Self::scalar_helper_expr(left, locals, proved, arities)
                    && Self::scalar_helper_expr(right, locals, proved, arities)
            }
            Expr::UnaryOp {
                op: UnaryOp::Neg | UnaryOp::Not,
                operand,
                ..
            } => Self::scalar_helper_expr(operand, locals, proved, arities),
            Expr::Call { func, args, .. } => {
                let Expr::Ident(name, _) = &**func else {
                    return false;
                };
                proved.contains(name)
                    && arities.get(name) == Some(&args.len())
                    && args
                        .iter()
                        .all(|arg| Self::scalar_helper_expr(arg, locals, proved, arities))
            }
            // These expressions may allocate, address memory, name globals,
            // use an unresolved type/callee, or contain hidden statements.
            Expr::StringLit(..)
            | Expr::GenericCall { .. }
            | Expr::Index { .. }
            | Expr::MemberAccess { .. }
            | Expr::Path { .. }
            | Expr::UnaryOp { .. }
            | Expr::BlockExpr(..)
            | Expr::SelfLit(..)
            | Expr::StructLit { .. }
            | Expr::ZeroInit(..) => false,
        }
    }

    fn scalar_helper_block(
        &self,
        block: &Block,
        outer: &HashSet<String>,
        proved: &HashSet<String>,
        arities: &BTreeMap<String, usize>,
    ) -> bool {
        // Lexical scope matters: a local inside one branch cannot authorize
        // an unbound/global identifier used after that branch.
        let mut locals = outer.clone();
        for stmt in &block.stmts {
            let pure_expr = |expr: &Expr| Self::scalar_helper_expr(expr, &locals, proved, arities);
            let pure = match stmt {
                Stmt::Let {
                    name,
                    ty,
                    init: Some(init),
                    cache_policy: None,
                    zero_drift: None,
                    bounds: None,
                    ..
                } => {
                    if !ty.as_ref().is_none_or(|ty| self.scalar_helper_type(ty)) || !pure_expr(init)
                    {
                        return false;
                    }
                    locals.insert(name.clone());
                    true
                }
                Stmt::Assign {
                    target: Expr::Ident(name, _),
                    value,
                    ..
                }
                | Stmt::CompoundAssign {
                    target: Expr::Ident(name, _),
                    value,
                    ..
                } => locals.contains(name) && pure_expr(value),
                Stmt::Expr(expr) | Stmt::Return(Some(expr), _) => pure_expr(expr),
                Stmt::If {
                    condition,
                    then_block,
                    else_block,
                    ..
                } => {
                    pure_expr(condition)
                        && self.scalar_helper_block(then_block, &locals, proved, arities)
                        && else_block.as_ref().is_none_or(|block| {
                            self.scalar_helper_block(block, &locals, proved, arities)
                        })
                }
                Stmt::While {
                    condition,
                    body,
                    invariant,
                    ..
                } => {
                    pure_expr(condition)
                        && invariant.as_ref().is_none_or(|expr| pure_expr(expr))
                        && self.scalar_helper_block(body, &locals, proved, arities)
                }
                Stmt::For {
                    loop_var,
                    start,
                    end,
                    step,
                    body,
                    invariant,
                    tile: None,
                    prefetch_stride: None,
                    ..
                } => {
                    if !pure_expr(start)
                        || !pure_expr(end)
                        || !step.as_ref().is_none_or(|expr| pure_expr(expr))
                    {
                        return false;
                    }
                    let mut loop_locals = locals.clone();
                    loop_locals.insert(loop_var.clone());
                    invariant.as_ref().is_none_or(|expr| {
                        Self::scalar_helper_expr(expr, &loop_locals, proved, arities)
                    }) && self.scalar_helper_block(body, &loop_locals, proved, arities)
                }
                Stmt::SafeBlock(body, _) => {
                    self.scalar_helper_block(body, &locals, proved, arities)
                }
                Stmt::Break { .. } => true,
                // In particular, refuse indirect stores, uninitialized or
                // non-scalar bindings, inline assembly and backend directives.
                Stmt::Let { .. }
                | Stmt::TypeAlias { .. }
                | Stmt::For { .. }
                | Stmt::Assign { .. }
                | Stmt::CompoundAssign { .. }
                | Stmt::Return(None, _)
                | Stmt::Chisel(..)
                | Stmt::Match { .. }
                | Stmt::GhostBlock(..)
                | Stmt::ClockDomainBlock { .. }
                | Stmt::CompileTimeAssert { .. }
                | Stmt::HintBlock { .. } => false,
            };
            if !pure {
                return false;
            }
        }
        true
    }

    /// Grow the set from scalar leaves to their proved callers. Recursion and
    /// unresolved source/runtime calls never enter the set. No call is moved
    /// or removed and no LLVM memory/effect attributes are asserted here.
    fn prove_scalar_runtime_helpers(&self, program: &Program) -> HashSet<String> {
        let mut proved = HashSet::new();
        if !self.optimize_helper_effects {
            return proved;
        }
        let mut functions = BTreeMap::new();
        let mut duplicate_names = HashSet::new();
        for item in &program.items {
            match item {
                Item::Func(function) => {
                    if functions.insert(function.name.clone(), function).is_some() {
                        duplicate_names.insert(function.name.clone());
                    }
                }
                Item::Impl(implementation) => {
                    for method in &implementation.methods {
                        duplicate_names
                            .insert(format!("{}_{}", implementation.target_type, method.name));
                    }
                }
                Item::Kernel(kernel) => {
                    duplicate_names.insert(kernel.name.clone());
                }
                _ => {}
            }
        }
        // `main` lowers to a different entry name. Do not summarize that
        // mapping or a top-level function that could collide with its body.
        if functions.contains_key("main") {
            duplicate_names.insert("ysu_main".into());
        }
        duplicate_names.insert("main".into());
        functions.retain(|name, _| {
            !duplicate_names.contains(name)
                // These names use special intrinsic dispatch even when a
                // source definition exists. Its scalar body therefore does
                // not establish the effects of the call actually emitted.
                && !matches!(
                    name.as_str(),
                    "load"
                        | "make_block_ptr2d"
                        | "block_ptr2d_load"
                        | "block_ptr2d_store"
                        | "block_ptr3d_load"
                        | "block_ptr3d_store"
                )
        });
        let arities = functions
            .iter()
            .map(|(name, function)| (name.clone(), function.params.len()))
            .collect();
        loop {
            let mut changed = false;
            for (name, function) in &functions {
                if proved.contains(name)
                    || !function.is_safe
                    || function.is_zk_safe
                    || function.is_zk_allow_unconstrained
                    || function.is_ptx_emit
                    || function.is_ghost
                    || function.is_hdl_emit
                    || function.tile.is_some()
                    || !function
                        .ret_ty
                        .as_ref()
                        .is_some_and(|ty| self.scalar_helper_type(ty))
                    || !function
                        .params
                        .iter()
                        .all(|param| self.scalar_helper_type(&param.ty))
                {
                    continue;
                }
                let locals = function
                    .params
                    .iter()
                    .map(|param| param.name.clone())
                    .collect();
                if self.scalar_helper_block(&function.body, &locals, &proved, &arities) {
                    proved.insert(name.clone());
                    changed = true;
                }
            }
            if !changed {
                return proved;
            }
        }
    }

    fn fresh_runtime_object(&self, expr: &Expr) -> Option<RuntimeObjectKind> {
        match expr {
            Expr::StringLit(..) if self.host_runtime_call("ystr_new") => {
                Some(RuntimeObjectKind::String)
            }
            Expr::Call { func, .. } => {
                let name = self.emit_call_target(func);
                if !self.host_runtime_call(&name) {
                    return None;
                }
                match name.as_str() {
                    "String_new"
                    | "String_clone"
                    | "ystr_new"
                    | "ystr_clone"
                    | "File_read_to_string"
                    | "yfile_read_to_string" => Some(RuntimeObjectKind::String),
                    "Vec_new" | "yvec_new" => Some(RuntimeObjectKind::Vector),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// A direct argument to these callbacks does not escape the header or its
    /// local slot. Mutators retain the header address; free clears the slot.
    fn runtime_argument_kind(name: &str, index: usize) -> Option<RuntimeObjectKind> {
        match (name, index) {
            (
                "String_new"
                | "String_clone"
                | "ystr_clone"
                | "String_len"
                | "ystr_len"
                | "String_char_at"
                | "ystr_char_at"
                | "String_push"
                | "ystr_push"
                | "String_free"
                | "ystr_free"
                | "File_read_to_string"
                | "yfile_read_to_string"
                | "print"
                | "println"
                | "yprint_str"
                | "yprintln_str"
                | "str_to_i64",
                0,
            )
            | (
                "String_eq" | "ystr_eq" | "String_eq_cstr" | "ystr_eq_cstr" | "String_push_str"
                | "ystr_push_str" | "File_write" | "yfile_write",
                0 | 1,
            ) => Some(RuntimeObjectKind::String),
            (
                "Vec_push" | "yvec_push" | "Vec_get" | "yvec_get" | "Vec_get_char"
                | "yvec_get_char" | "Vec_len" | "yvec_len" | "Vec_free" | "yvec_free",
                0,
            ) => Some(RuntimeObjectKind::Vector),
            _ => None,
        }
    }

    fn inspect_runtime_expr(
        &self,
        expr: &Expr,
        candidates: &BTreeMap<String, RuntimeObjectKind>,
        rejected: &mut HashSet<String>,
        opaque: &mut bool,
    ) {
        match expr {
            Expr::Ident(name, _) => {
                if candidates.contains_key(name) {
                    rejected.insert(name.clone());
                }
            }
            Expr::Call { func, args, .. } => {
                let name = self.emit_call_target(func);
                let host_call = self.host_runtime_call(&name);
                // Even an unknown call without an object argument can invoke
                // code with hidden state. Keep the whole function opaque.
                if !host_call
                    && !self.scalar_runtime_helpers.contains(&name)
                    && !(name == "load" && !self.fn_llvm_params.contains_key(&name))
                {
                    *opaque = true;
                }
                let freeing = host_call
                    && matches!(
                        name.as_str(),
                        "String_free" | "ystr_free" | "Vec_free" | "yvec_free"
                    );
                for (index, arg) in args.iter().enumerate() {
                    let protected = if host_call {
                        Self::runtime_arg_local(arg).and_then(|local| {
                            let kind = candidates.get(local)?;
                            (Some(*kind) == Self::runtime_argument_kind(&name, index))
                                .then_some(local)
                        })
                    } else {
                        None
                    };
                    if let Some(local) = protected {
                        // Direct-handle free leaves the caller's slot stale.
                        // Only a direct mutable slot reference is proven safe.
                        if freeing
                            && !matches!(
                                arg,
                                Expr::UnaryOp {
                                    op: UnaryOp::Ref { mutable: true },
                                    ..
                                }
                            )
                        {
                            rejected.insert(local.to_string());
                        }
                    } else {
                        self.inspect_runtime_expr(arg, candidates, rejected, opaque);
                    }
                }
            }
            Expr::GenericCall { args, .. } => {
                *opaque = true;
                for arg in args {
                    self.inspect_runtime_expr(arg, candidates, rejected, opaque);
                }
            }
            Expr::Index { base, index, .. }
            | Expr::BinaryOp {
                left: base,
                right: index,
                ..
            } => {
                self.inspect_runtime_expr(base, candidates, rejected, opaque);
                self.inspect_runtime_expr(index, candidates, rejected, opaque);
            }
            Expr::MemberAccess { base, .. } | Expr::UnaryOp { operand: base, .. } => {
                self.inspect_runtime_expr(base, candidates, rejected, opaque);
            }
            Expr::StructLit { fields, .. } => {
                for (_, value) in fields {
                    self.inspect_runtime_expr(value, candidates, rejected, opaque);
                }
            }
            // A statement-bearing argument can free a slot between capturing
            // a direct argument and callback resolution. Keep those functions
            // opaque rather than assuming an evaluation/lifetime ordering.
            Expr::BlockExpr(..) => *opaque = true,
            Expr::IntLit(..)
            | Expr::FloatLit(..)
            | Expr::StringLit(..)
            | Expr::CharLit(..)
            | Expr::Path { .. }
            | Expr::BoolLit(..)
            | Expr::SelfLit(..)
            | Expr::ZeroInit(..) => {}
        }
    }

    /// Prove closed ownership for a whole renamed function before emitting any
    /// reads. This handles loops and branches without optimistic flow merges:
    /// assignments, aliases, escapes and opaque calls rule out the fast path.
    fn prove_runtime_locals(&self, function: &FuncDecl) -> BTreeMap<String, RuntimeObjectKind> {
        if !self.optimize_runtime && !self.optimize_runtime_mutations {
            return BTreeMap::new();
        }
        let program = Program {
            items: vec![Item::Func(function.clone())],
        };
        let mut candidates = BTreeMap::new();
        crate::ast::for_each_stmt(&program, &mut |stmt| {
            if let Stmt::Let {
                name,
                init: Some(init),
                ..
            } = stmt
            {
                if let Some(kind) = self.fresh_runtime_object(init) {
                    candidates.insert(name.clone(), kind);
                }
            }
        });
        if candidates.is_empty() {
            return candidates;
        }
        let mut rejected = HashSet::new();
        let mut opaque = false;
        crate::ast::for_each_stmt(&program, &mut |stmt| {
            let mut inspect = |expr: &Expr| {
                self.inspect_runtime_expr(expr, &candidates, &mut rejected, &mut opaque)
            };
            match stmt {
                Stmt::Let { init, bounds, .. } => {
                    if let Some(init) = init {
                        inspect(init);
                    }
                    if let Some(bounds) = bounds {
                        inspect(&bounds.min);
                        inspect(&bounds.max);
                    }
                }
                Stmt::Assign { target, value, .. } | Stmt::CompoundAssign { target, value, .. } => {
                    inspect(target);
                    inspect(value);
                }
                Stmt::Expr(expr) | Stmt::Return(Some(expr), _) => inspect(expr),
                Stmt::If { condition, .. } | Stmt::CompileTimeAssert { condition, .. } => {
                    inspect(condition)
                }
                Stmt::While {
                    condition,
                    invariant,
                    ..
                } => {
                    inspect(condition);
                    if let Some(invariant) = invariant {
                        inspect(invariant);
                    }
                }
                Stmt::For {
                    start,
                    end,
                    step,
                    invariant,
                    tile,
                    prefetch_stride,
                    ..
                } => {
                    inspect(start);
                    inspect(end);
                    if let Some(step) = step {
                        inspect(step);
                    }
                    if let Some(invariant) = invariant {
                        inspect(invariant);
                    }
                    if let Some(tile) = tile {
                        inspect(&tile.block_m);
                        inspect(&tile.block_n);
                        if let Some(k) = &tile.block_k {
                            inspect(k);
                        }
                    }
                    if let Some(stride) = prefetch_stride
                        .as_ref()
                        .and_then(|attr| attr.stride.as_ref())
                    {
                        inspect(stride);
                    }
                }
                Stmt::Match {
                    scrutinee, arms, ..
                } => {
                    inspect(scrutinee);
                    for arm in arms {
                        if let MatchPattern::Literal(expr) = &arm.pattern {
                            inspect(expr);
                        }
                        inspect(&arm.body);
                    }
                }
                Stmt::ClockDomainBlock { clock, .. } => inspect(clock),
                Stmt::Chisel(..) => opaque = true,
                _ => {}
            }
        });
        if opaque {
            candidates.clear();
        } else {
            candidates.retain(|name, _| !rejected.contains(name));
        }
        candidates
    }

    fn prove_runtime_vector_sizes(&self, function: &FuncDecl) -> BTreeMap<String, u64> {
        let mut sizes = BTreeMap::new();
        if !self.optimize_runtime_mutations {
            return sizes;
        }
        let program = Program {
            items: vec![Item::Func(function.clone())],
        };
        crate::ast::for_each_stmt(&program, &mut |stmt| {
            let Stmt::Let {
                name,
                init: Some(Expr::Call { func, args, .. }),
                ..
            } = stmt
            else {
                return;
            };
            if self.runtime_locals.get(name) != Some(&RuntimeObjectKind::Vector) || args.len() != 1
            {
                return;
            }
            let Expr::IntLit(value, _) = &args[0] else {
                return;
            };
            let size = match self.emit_call_target(func).as_str() {
                // Vec_new truncates to i32 before its native wrapper widens.
                "Vec_new" => i64::from(*value as i32),
                "yvec_new" => *value,
                _ => return,
            };
            if size > 0 {
                sizes.insert(name.clone(), size as u64);
            }
        });
        sizes
    }

    fn runtime_element_extent(&self, expression: &Expr, implicit_lvalue: bool) -> Option<u64> {
        // Only direct scalar stack storage is proved readable here. Borrowed
        // pointers (including a vector's own data) stay on the callback path.
        let local = match expression {
            Expr::UnaryOp {
                op: UnaryOp::Ref { .. },
                operand,
                ..
            } => {
                let Expr::Ident(local, _) = &**operand else {
                    return None;
                };
                local
            }
            // Vec_push takes the lvalue's address even without an explicit
            // reference. Raw yvec_push instead receives the expression value.
            Expr::Ident(local, _) if implicit_lvalue => local,
            _ => return None,
        };
        match self.locals.get(local)?.as_str() {
            "i1" | "i8" => Some(1),
            "i16" | "half" => Some(2),
            "i32" | "float" => Some(4),
            "i64" | "double" | "ptr" => Some(8),
            _ => None,
        }
    }

    fn try_emit_runtime_append(&mut self, name: &str, args: &[Expr]) -> Option<String> {
        if !self.optimize_runtime_mutations || !self.host_runtime_call(name) || args.len() != 2 {
            return None;
        }
        let kind = match name {
            "String_push" | "ystr_push" => RuntimeObjectKind::String,
            "Vec_push" | "yvec_push" => RuntimeObjectKind::Vector,
            _ => return None,
        };
        let local = Self::runtime_arg_local(&args[0])?;
        if self.runtime_locals.get(local) != Some(&kind)
            || self.locals.get(local).map(String::as_str) != Some("ptr")
        {
            return None;
        }
        let element_size = if kind == RuntimeObjectKind::Vector {
            let extent = self.runtime_element_extent(&args[1], name == "Vec_push")?;
            let size = match self.runtime_vector_sizes.get(local) {
                Some(size) => *size,
                // A dynamic element size is safe only when its header field
                // exactly matches this proved scalar source storage extent.
                // The guard below keeps the copy length a constant, and all
                // mismatched widths retain the original callback.
                None if self.optimize_runtime_copies => extent,
                None => return None,
            };
            if size > extent {
                return None;
            }
            Some(size)
        } else {
            None
        };
        // Capture the original call arguments once, in their original order.
        let first = self.emit_expr(&args[0], None, None);
        let second = if kind == RuntimeObjectKind::String {
            let value = self.emit_expr(&args[1], None, None);
            let ty = self.infer_type(&args[1]);
            let unsigned = self.expr_is_unsigned(&args[1]);
            self.emit_coerce_from(&value, &ty, "i8", unsigned)
        } else if name == "Vec_push" {
            self.emit_lvalue(&args[1])
        } else {
            self.emit_expr(&args[1], None, None)
        };
        let handle = if matches!(
            &args[0],
            Expr::UnaryOp {
                op: UnaryOp::Ref { .. },
                ..
            }
        ) {
            self.emit_load(&first, "ptr")
        } else {
            first.clone()
        };
        let read = self.fresh_label("runtime.append.read");
        let fast = self.fresh_label("runtime.append.fast");
        let slow = self.fresh_label("runtime.append.slow");
        let merge = self.fresh_label("runtime.append.merge");
        let header = if kind == RuntimeObjectKind::String {
            "{ ptr, i64, i64 }"
        } else {
            "{ ptr, i64, i64, i64 }"
        };
        self.wln("  ; CPU JIT guarded runtime append (closed local allocation)");
        let live = self.fresh_tmp();
        writeln!(&mut self.output, "  {live} = icmp ne ptr {handle}, null").unwrap();
        writeln!(
            &mut self.output,
            "  br i1 {live}, label %{read}, label %{slow}\n{read}:"
        )
        .unwrap();
        let len_ptr = self.fresh_tmp();
        let cap_ptr = self.fresh_tmp();
        let data_ptr = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {len_ptr} = getelementptr {header}, ptr {handle}, i32 0, i32 1"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  {cap_ptr} = getelementptr {header}, ptr {handle}, i32 0, i32 2"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  {data_ptr} = getelementptr {header}, ptr {handle}, i32 0, i32 0"
        )
        .unwrap();
        let len = self.emit_load(&len_ptr, "i64");
        let cap = self.emit_load(&cap_ptr, "i64");
        let data = self.emit_load(&data_ptr, "ptr");
        let nonnegative = self.fresh_tmp();
        let data_live = self.fresh_tmp();
        writeln!(&mut self.output, "  {nonnegative} = icmp sge i64 {len}, 0").unwrap();
        writeln!(&mut self.output, "  {data_live} = icmp ne ptr {data}, null").unwrap();
        let mut checks = vec![nonnegative, data_live];
        if let Some(size) = element_size {
            let spare = self.fresh_tmp();
            let src_live = self.fresh_tmp();
            let size_ptr = self.fresh_tmp();
            writeln!(&mut self.output, "  {spare} = icmp slt i64 {len}, {cap}").unwrap();
            writeln!(
                &mut self.output,
                "  {src_live} = icmp ne ptr {second}, null"
            )
            .unwrap();
            writeln!(
                &mut self.output,
                "  {size_ptr} = getelementptr {header}, ptr {handle}, i32 0, i32 3"
            )
            .unwrap();
            let actual_size = self.emit_load(&size_ptr, "i64");
            let size_matches = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {size_matches} = icmp eq i64 {actual_size}, {size}"
            )
            .unwrap();
            checks.extend([spare, src_live, size_matches]);
        } else {
            // cap includes the trailing NUL. Avoid len+2 overflow entirely.
            let usable = self.fresh_tmp();
            let capacity_valid = self.fresh_tmp();
            let spare = self.fresh_tmp();
            writeln!(&mut self.output, "  {usable} = sub i64 {cap}, 2").unwrap();
            writeln!(
                &mut self.output,
                "  {capacity_valid} = icmp sge i64 {cap}, 2"
            )
            .unwrap();
            writeln!(&mut self.output, "  {spare} = icmp sle i64 {len}, {usable}").unwrap();
            checks.extend([capacity_valid, spare]);
        }
        let mut allowed = checks.remove(0);
        for check in checks {
            let both = self.fresh_tmp();
            writeln!(&mut self.output, "  {both} = and i1 {allowed}, {check}").unwrap();
            allowed = both;
        }
        writeln!(
            &mut self.output,
            "  br i1 {allowed}, label %{fast}, label %{slow}\n{fast}:"
        )
        .unwrap();
        let next_len = self.fresh_tmp();
        writeln!(&mut self.output, "  {next_len} = add i64 {len}, 1").unwrap();
        if let Some(size) = element_size {
            let offset = self.fresh_tmp();
            let destination = self.fresh_tmp();
            writeln!(&mut self.output, "  {offset} = mul i64 {len}, {size}").unwrap();
            writeln!(
                &mut self.output,
                "  {destination} = getelementptr i8, ptr {data}, i64 {offset}"
            )
            .unwrap();
            self.called_functions.push("llvm.memmove.p0.p0.i64".into());
            writeln!(&mut self.output, "  call void @llvm.memmove.p0.p0.i64(ptr align 1 {destination}, ptr align 1 {second}, i64 {size}, i1 false)").unwrap();
        } else {
            let destination = self.fresh_tmp();
            let terminator = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {destination} = getelementptr i8, ptr {data}, i64 {len}"
            )
            .unwrap();
            writeln!(
                &mut self.output,
                "  store i8 {second}, ptr {destination}, align 1"
            )
            .unwrap();
            writeln!(
                &mut self.output,
                "  {terminator} = getelementptr i8, ptr {data}, i64 {next_len}"
            )
            .unwrap();
            writeln!(&mut self.output, "  store i8 0, ptr {terminator}, align 1").unwrap();
        }
        writeln!(
            &mut self.output,
            "  store i64 {next_len}, ptr {len_ptr}, align 8\n  br label %{merge}\n{slow}:"
        )
        .unwrap();
        self.called_functions.push(name.to_string());
        let second_type = if kind == RuntimeObjectKind::String {
            "i8"
        } else {
            "ptr"
        };
        writeln!(&mut self.output, "  call void @{name}(ptr {first}, {second_type} {second})\n  br label %{merge}\n{merge}:").unwrap();
        Some(self.fresh_tmp().replace("%_t", "%_void"))
    }

    fn try_emit_runtime_bulk_append(&mut self, name: &str, args: &[Expr]) -> Option<String> {
        if !self.optimize_runtime_mutations
            || !self.optimize_runtime_copies
            || !self.host_runtime_call(name)
            || !matches!(name, "String_push_str" | "ystr_push_str")
            || args.len() != 2
        {
            return None;
        }
        for arg in args {
            let local = Self::runtime_arg_local(arg)?;
            if self.runtime_locals.get(local) != Some(&RuntimeObjectKind::String)
                || self.locals.get(local).map(String::as_str) != Some("ptr")
            {
                return None;
            }
        }
        // Calls receive each expression once, left to right. A reference
        // captures its slot address; resolve both slots after evaluation.
        let first = self.emit_expr(&args[0], None, None);
        let second = self.emit_expr(&args[1], None, None);
        let handle = |emitter: &mut Self, argument: &Expr, value: &str| {
            if matches!(
                argument,
                Expr::UnaryOp {
                    op: UnaryOp::Ref { .. },
                    ..
                }
            ) {
                emitter.emit_load(value, "ptr")
            } else {
                value.to_string()
            }
        };
        let destination_handle = handle(self, &args[0], &first);
        let source_handle = handle(self, &args[1], &second);
        let read = self.fresh_label("runtime.copy.read");
        let fast = self.fresh_label("runtime.copy.fast");
        let slow = self.fresh_label("runtime.copy.slow");
        let merge = self.fresh_label("runtime.copy.merge");
        self.wln("  ; CPU JIT guarded bulk String append (closed local allocations)");
        let destination_live = self.fresh_tmp();
        let source_live = self.fresh_tmp();
        let both_live = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {destination_live} = icmp ne ptr {destination_handle}, null"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  {source_live} = icmp ne ptr {source_handle}, null"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  {both_live} = and i1 {destination_live}, {source_live}"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  br i1 {both_live}, label %{read}, label %{slow}\n{read}:"
        )
        .unwrap();
        let header = "{ ptr, i64, i64 }";
        let field = |emitter: &mut Self, handle: &str, index: u32, ty: &str| {
            let pointer = emitter.fresh_tmp();
            writeln!(
                &mut emitter.output,
                "  {pointer} = getelementptr {header}, ptr {handle}, i32 0, i32 {index}"
            )
            .unwrap();
            let value = emitter.emit_load(&pointer, ty);
            (pointer, value)
        };
        let (destination_len_pointer, destination_len) = field(self, &destination_handle, 1, "i64");
        let (_, destination_cap) = field(self, &destination_handle, 2, "i64");
        let (_, destination_data) = field(self, &destination_handle, 0, "ptr");
        // Snapshot source length and data before any destination mutation.
        // They may belong to the same header during self-append.
        let (_, source_len) = field(self, &source_handle, 1, "i64");
        let (_, source_data) = field(self, &source_handle, 0, "ptr");
        let destination_nonnegative = self.fresh_tmp();
        let source_nonnegative = self.fresh_tmp();
        let positive_capacity = self.fresh_tmp();
        let within_capacity = self.fresh_tmp();
        let destination_data_live = self.fresh_tmp();
        let source_data_live = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {destination_nonnegative} = icmp sge i64 {destination_len}, 0"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  {source_nonnegative} = icmp sge i64 {source_len}, 0"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  {positive_capacity} = icmp sgt i64 {destination_cap}, 0"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  {within_capacity} = icmp slt i64 {destination_len}, {destination_cap}"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  {destination_data_live} = icmp ne ptr {destination_data}, null"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  {source_data_live} = icmp ne ptr {source_data}, null"
        )
        .unwrap();
        // Plain subtraction has no poison-producing flags. These values are
        // used only when the guards prove 0 <= len < cap and cap > 0.
        let available = self.fresh_tmp();
        let remaining = self.fresh_tmp();
        let fits = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {available} = sub i64 {destination_cap}, {destination_len}"
        )
        .unwrap();
        writeln!(&mut self.output, "  {remaining} = sub i64 {available}, 1").unwrap();
        writeln!(
            &mut self.output,
            "  {fits} = icmp sle i64 {source_len}, {remaining}"
        )
        .unwrap();
        let mut allowed = destination_nonnegative;
        for check in [
            source_nonnegative,
            positive_capacity,
            within_capacity,
            destination_data_live,
            source_data_live,
            fits,
        ] {
            let both = self.fresh_tmp();
            writeln!(&mut self.output, "  {both} = and i1 {allowed}, {check}").unwrap();
            allowed = both;
        }
        writeln!(
            &mut self.output,
            "  br i1 {allowed}, label %{fast}, label %{slow}\n{fast}:"
        )
        .unwrap();
        let destination = self.fresh_tmp();
        let next_len = self.fresh_tmp();
        let terminator = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {destination} = getelementptr i8, ptr {destination_data}, i64 {destination_len}"
        )
        .unwrap();
        self.called_functions.push("llvm.memmove.p0.p0.i64".into());
        writeln!(&mut self.output, "  call void @llvm.memmove.p0.p0.i64(ptr align 1 {destination}, ptr align 1 {source_data}, i64 {source_len}, i1 false)").unwrap();
        writeln!(
            &mut self.output,
            "  {next_len} = add i64 {destination_len}, {source_len}"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  store i64 {next_len}, ptr {destination_len_pointer}, align 8"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  {terminator} = getelementptr i8, ptr {destination_data}, i64 {next_len}"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  store i8 0, ptr {terminator}, align 1\n  br label %{merge}\n{slow}:"
        )
        .unwrap();
        self.called_functions.push(name.to_string());
        writeln!(
            &mut self.output,
            "  call void @{name}(ptr {first}, ptr {second})\n  br label %{merge}\n{merge}:"
        )
        .unwrap();
        Some(self.fresh_tmp().replace("%_t", "%_void"))
    }

    fn try_emit_runtime_query(&mut self, name: &str, args: &[Expr]) -> Option<String> {
        if !self.optimize_runtime || !self.host_runtime_call(name) {
            return None;
        }
        if name == "ychar_to_ascii" && args.len() == 1 {
            let value = self.emit_expr(&args[0], None, None);
            let ty = self.infer_type(&args[0]);
            let unsigned = self.expr_is_unsigned(&args[0]);
            let byte = self.emit_coerce_from(&value, &ty, "i8", unsigned);
            let result = self.fresh_tmp();
            // The runtime receives u8 and returns i32::from(byte).
            writeln!(&mut self.output, "  {result} = zext i8 {byte} to i32").unwrap();
            return Some(result);
        }
        let (kind, result_type, indexed) = match name {
            "String_len" | "ystr_len" => (RuntimeObjectKind::String, "i64", false),
            "Vec_len" | "yvec_len" => (RuntimeObjectKind::Vector, "i64", false),
            "String_char_at" | "ystr_char_at" => (RuntimeObjectKind::String, "i8", true),
            "Vec_get_char" | "yvec_get_char" => (RuntimeObjectKind::Vector, "i8", true),
            "Vec_get" | "yvec_get" => (RuntimeObjectKind::Vector, "ptr", true),
            _ => return None,
        };
        if args.len() != if indexed { 2 } else { 1 } {
            return None;
        }
        let local = Self::runtime_arg_local(&args[0])?;
        if self.runtime_locals.get(local) != Some(&kind)
            || self.locals.get(local).map(String::as_str) != Some("ptr")
        {
            return None;
        }
        let slot = format!("%{local}");
        let reference = matches!(
            &args[0],
            Expr::UnaryOp {
                op: UnaryOp::Ref { .. },
                ..
            }
        );
        let captured = if reference {
            None
        } else {
            Some(self.emit_load(&slot, "ptr"))
        };
        // Preserve argument evaluation even if the handle is null.
        let index = if indexed {
            let value = self.emit_expr(&args[1], None, None);
            let ty = self.infer_type(&args[1]);
            let unsigned = self.expr_is_unsigned(&args[1]);
            Some(self.emit_coerce_from(&value, &ty, "i64", unsigned))
        } else {
            None
        };
        // A reference argument captures the slot address. Its callback loads
        // the handle after evaluating later arguments; preserve that order.
        let handle = captured.unwrap_or_else(|| self.emit_load(&slot, "ptr"));
        let read = self.fresh_label("runtime.read");
        let valid = self.fresh_label("runtime.valid");
        let neutral = self.fresh_label("runtime.neutral");
        let merge = self.fresh_label("runtime.merge");
        let header = match kind {
            RuntimeObjectKind::String => "{ ptr, i64, i64 }",
            RuntimeObjectKind::Vector => "{ ptr, i64, i64, i64 }",
        };
        self.wln("  ; CPU JIT runtime header read (closed local allocation)");
        let nonnull = self.fresh_tmp();
        writeln!(&mut self.output, "  {nonnull} = icmp ne ptr {handle}, null").unwrap();
        writeln!(
            &mut self.output,
            "  br i1 {nonnull}, label %{read}, label %{neutral}"
        )
        .unwrap();
        writeln!(&mut self.output, "{read}:").unwrap();
        let length_pointer = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {length_pointer} = getelementptr {header}, ptr {handle}, i32 0, i32 1"
        )
        .unwrap();
        let length = self.emit_load(&length_pointer, "i64");
        let (value, value_block) = if let Some(index) = index {
            let nonnegative = self.fresh_tmp();
            let below_length = self.fresh_tmp();
            let in_bounds = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {nonnegative} = icmp sge i64 {index}, 0"
            )
            .unwrap();
            writeln!(
                &mut self.output,
                "  {below_length} = icmp slt i64 {index}, {length}"
            )
            .unwrap();
            writeln!(
                &mut self.output,
                "  {in_bounds} = and i1 {nonnegative}, {below_length}"
            )
            .unwrap();
            writeln!(
                &mut self.output,
                "  br i1 {in_bounds}, label %{valid}, label %{neutral}"
            )
            .unwrap();
            writeln!(&mut self.output, "{valid}:").unwrap();
            let data_pointer = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {data_pointer} = getelementptr {header}, ptr {handle}, i32 0, i32 0"
            )
            .unwrap();
            let data = self.emit_load(&data_pointer, "ptr");
            let offset = if kind == RuntimeObjectKind::Vector {
                let size_pointer = self.fresh_tmp();
                writeln!(
                    &mut self.output,
                    "  {size_pointer} = getelementptr {header}, ptr {handle}, i32 0, i32 3"
                )
                .unwrap();
                let size = self.emit_load(&size_pointer, "i64");
                let offset = self.fresh_tmp();
                writeln!(&mut self.output, "  {offset} = mul i64 {index}, {size}").unwrap();
                offset
            } else {
                index
            };
            let element = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {element} = getelementptr i8, ptr {data}, i64 {offset}"
            )
            .unwrap();
            let value = if result_type == "ptr" {
                element
            } else {
                self.emit_load(&element, "i8")
            };
            writeln!(&mut self.output, "  br label %{merge}").unwrap();
            (value, valid)
        } else {
            writeln!(&mut self.output, "  br label %{merge}").unwrap();
            (length, read)
        };
        writeln!(
            &mut self.output,
            "{neutral}:\n  br label %{merge}\n{merge}:"
        )
        .unwrap();
        let result = self.fresh_tmp();
        let neutral_value = if result_type == "ptr" { "null" } else { "0" };
        writeln!(&mut self.output, "  {result} = phi {result_type} [{value}, %{value_block}], [{neutral_value}, %{neutral}]").unwrap();
        Some(result)
    }

    fn emit_func(&mut self, f: &FuncDecl) {
        // Every binding gets a name - and so a slot - of its own: this backend
        // keeps one slot per name per function, and without the renaming a
        // nested or repeated `let` wrote the slot of the binding it shadows
        // (`crate::lexical_scope`). A function that binds no name twice is
        // unchanged.
        let renamed = FuncDecl {
            body: crate::lexical_scope::unique_bindings(&f.params, &f.body),
            ..f.clone()
        };
        let f = &renamed;
        self.reset_function_state();
        self.current_ret_q = match &f.ret_ty {
            Some(Type::Primitive(n, _)) | Some(Type::Ident(n, _)) => QFormat::parse(n),
            _ => None,
        };
        self.runtime_locals = self.prove_runtime_locals(f);
        self.runtime_vector_sizes = self.prove_runtime_vector_sizes(f);
        let prev_ptx = self.in_ptx_emit;
        self.in_ptx_emit = f.is_ptx_emit;

        let ret_type = match &f.ret_ty {
            Some(ty) => self.emit_type(ty),
            None if self.aot_entry_status && f.name == "main" && self.current_impl_target.is_none() => "i32".into(),
            None => "void".into(),
        };
        // An array evaluates to its storage's address, and a local's storage
        // is this function's own stack frame: `return v;` would hand the
        // caller a pointer into a frame that no longer exists. (It returned 0
        // where the answer was 11 before local arrays had storage at all.)
        if let Some(t @ Type::Array { .. }) = &f.ret_ty {
            self.emit_errors.push(format!(
                "Line {}: [LLVM host backend] `fn {}` returns an array (`{}`); this backend \
                 would return the address of storage in the callee's own stack frame. Return a \
                 struct with an array field instead - a struct is returned by value.",
                f.span.line,
                f.name,
                ast_type_to_string(t)
            ));
        }

        let func_name = if let Some(ref target) = self.current_impl_target {
            format!("{}_{}", target, f.name)
        } else if f.name == "main" {
            "ysu_main".to_string()
        } else {
            f.name.clone()
        };
        self.defined_functions.push(func_name.clone());
        if let Some(d) = &mut self.debug {
            // `fn main` is emitted as `ysu_main` because the C runtime owns the
            // process's `main`; the debugger still calls it `main`.
            let display = if func_name == "ysu_main" {
                "main"
            } else {
                func_name.as_str()
            };
            d.begin_function(
                &func_name,
                display,
                f.span.line,
                f.span.col,
                &f.params,
                f.ret_ty.as_ref(),
            );
            if let Some(target) = &self.current_impl_target {
                d.set_method(&func_name, &format!("{}::{}", target, f.name));
            }
        }

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
            self.emit_param_slot(p);
        }

        writeln!(
            &mut self.output,
            "  {} = alloca [8 x i8], align 8",
            Y_OOB_SINK
        )
        .unwrap();

        // Forward declare all lets in entry block to avoid loop stack growth
        self.emit_alloca_for_block(&f.body);

        self.emit_block_body(&f.body, &ret_type);

        // The implicit return belongs to the body's last statement under `-g`.
        // Left where `emit_stmt` restores to, the function's own line, `next`
        // from the last statement would stop on the `fn` header on the way out.
        let outer = match f.body.stmts.last() {
            Some(last) if !self.block_terminated => self.dbg_enter(&last.span()),
            _ => None,
        };
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
        self.dbg_leave(outer);

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
                Stmt::Let {
                    name,
                    ty,
                    zero_drift: Some(_),
                    bounds,
                    span,
                    ..
                } if !self.locals.contains_key(name) => {
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
                                crate::lexical_scope::source_name(name),
                                &ty_name,
                                &decision,
                                crate::zero_drift::explain_requested(),
                            ));
                            self.locals
                                .insert(name.clone(), decision.repr.llvm_type().to_string());
                            self.locals_ast_type.insert(name.clone(), ty_name.clone());
                            let integer_domain = decision.repr.frac_bits() == 0
                                && matches!(
                                    ty_name.as_str(),
                                    "I8" | "I16"
                                        | "I32"
                                        | "I64"
                                        | "U8"
                                        | "U16"
                                        | "U32"
                                        | "U64"
                                        | "i8"
                                        | "i16"
                                        | "i32"
                                        | "i64"
                                        | "u8"
                                        | "u16"
                                        | "u32"
                                        | "u64"
                                        | "isize"
                                        | "usize"
                                );
                            self.zero_drift
                                .insert(name.clone(), (decision.repr, integer_domain));
                            writeln!(
                                &mut self.output,
                                "  %{} = alloca {}",
                                name,
                                decision.repr.llvm_type()
                            )
                            .unwrap();
                            if self.debug.is_some() {
                                // The slot holds the accumulator's exact
                                // representation, not the declared value: a
                                // Q format is the value times 2^frac, and the
                                // type's name says so.
                                let repr = decision.repr;
                                let dty = crate::debug_info::DbgTy::Int {
                                    name: if repr.frac_bits() == 0 {
                                        repr.name().to_string()
                                    } else {
                                        format!("{}_raw", repr.name())
                                    },
                                    bits: repr.total_bits() as u64,
                                    signed: true,
                                };
                                let d = self.debug.as_mut().unwrap();
                                if let Some(i) =
                                    d.declare(name, span.line, span.col, false, Some(dty))
                                {
                                    writeln!(
                                        &mut self.output,
                                        "{}{}",
                                        crate::debug_info::VAR_MARKER,
                                        i
                                    )
                                    .unwrap();
                                }
                            }
                        }
                        Err(why) => {
                            self.emit_errors.push(format!(
                                "Line {}: @ZeroDrift on `{}: {}` cannot be honoured. No exact \
representation holds that range at that resolution, and only exact (integer or fixed-point) \
accumulation is drift-free - f64 is the same non-associative arithmetic with more mantissa. \
Add @bounds(min, max) to state the accumulator's real range, or declare it as a Q format.\n{}",
                                span.line,
                                crate::lexical_scope::source_name(name),
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
                Stmt::Let {
                    name,
                    ty,
                    init,
                    span,
                    ..
                } => {
                    // A local array is given storage of its own - see
                    // `local_array_type` for what it used to be given.
                    let array_ty = match ty {
                        Some(t @ Type::Array { .. }) => Some(
                            self.local_array_type(t, "a local array", span)
                                .unwrap_or_else(|| "ptr".into()),
                        ),
                        _ => None,
                    };
                    // Every binding has a name of its own by now
                    // (`emit_func` renames them apart), so this is the first
                    // time `name` is seen: a second `let` of a source name is
                    // `name.1`, with a slot of its own of whatever size it
                    // declares.
                    if !self.locals.contains_key(name) {
                        let ir_ty = match ty {
                            Some(t) => {
                                if let Some(pty) = self.get_pointee_type(t) {
                                    self.pointee_types.insert(name.clone(), pty);
                                }
                                match &array_ty {
                                    Some(aty) => aty.clone(),
                                    None => self.emit_type(t),
                                }
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
                        // Aggregate copies here are `memcpy` with `align 8` on
                        // both pointers, so an array's storage must honour it.
                        let align = if ir_ty.starts_with('[') {
                            ", align 8"
                        } else {
                            ""
                        };
                        writeln!(&mut self.output, "  %{} = alloca {}{}", name, ir_ty, align)
                            .unwrap();
                        self.dbg_var(name, span, ty.as_ref(), init.as_ref(), &ir_ty, false);
                    }
                }
                Stmt::For {
                    loop_var,
                    body,
                    span,
                    ..
                } => {
                    self.locals.insert(loop_var.clone(), "i32".into());
                    writeln!(&mut self.output, "  %{} = alloca i32", loop_var).unwrap();
                    self.dbg_var(loop_var, span, None, None, "i32", false);
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

    /// A parameter's named slot, for `emit_func` and `emit_kernel` alike.
    ///
    /// An array arrives as a pointer - the signature the definition and every
    /// call site agree on - and is COPIED into storage of its own. Arrays are
    /// values: the ZK backend binds each element separately, and `--emit-cpu`
    /// prints Rust's by-value `[T; N]`, so a callee writing `a[0]` must not
    /// write its caller's array. The type checker allows that write.
    fn emit_param_slot(&mut self, p: &Param) {
        self.locals_ast_type
            .insert(p.name.clone(), ast_type_to_string(&p.ty));
        if let Type::Array { .. } = &p.ty {
            if let Some(aty) = self.local_array_type(&p.ty, "a parameter", &p.span) {
                self.locals.insert(p.name.clone(), aty.clone());
                writeln!(&mut self.output, "  %{} = alloca {}, align 8", p.name, aty).unwrap();
                let size = self.emit_sizeof(&aty);
                writeln!(
                    &mut self.output,
                    "  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %{}, ptr align 1 %{}.arg, i64 {}, i1 false)",
                    p.name, p.name, size
                )
                .unwrap();
                self.dbg_var(&p.name, &p.span, Some(&p.ty), None, &aty, true);
                return;
            }
        }
        let ty = self.emit_type(&p.ty);
        self.locals.insert(p.name.clone(), ty.clone());
        if let Some(pty) = self.get_pointee_type(&p.ty) {
            self.pointee_types.insert(p.name.clone(), pty);
        }
        if let Some(ety) = memory_element_llvm_type(&p.ty) {
            self.mem_elem_types.insert(p.name.clone(), ety);
        }
        if let Some(sty) = memory_storage_llvm_type(&p.ty) {
            self.mem_storage_types.insert(p.name.clone(), sty);
        }
        if let Type::Generic { base, args, .. } = &p.ty {
            if matches!(base.as_str(), "GlobalMemory" | "SharedMemory") {
                if let Some(GenericArg::Type(element)) = args.first() {
                    self.mem_ast_types
                        .insert(p.name.clone(), ast_type_to_string(element));
                }
            }
        }
        writeln!(&mut self.output, "  %{} = alloca {}", p.name, ty).unwrap();
        self.emit_store(&format!("%{}.arg", p.name), &format!("%{}", p.name), &ty);
        self.dbg_var(&p.name, &p.span, Some(&p.ty), None, &ty, true);
    }

    /// `sizeof(ty)` in bytes, by the `getelementptr ty, ptr null, 1` idiom.
    fn emit_sizeof(&mut self, ty: &str) -> String {
        let end = self.fresh_tmp();
        let size = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {} = getelementptr {}, ptr null, i32 1",
            end, ty
        )
        .unwrap();
        writeln!(&mut self.output, "  {} = ptrtoint ptr {} to i64", size, end).unwrap();
        size
    }

    fn emit_kernel(&mut self, k: &KernelDecl) {
        // As in `emit_func`: a binding of its own for every `let`.
        let renamed = KernelDecl {
            body: crate::lexical_scope::unique_bindings(&k.params, &k.body),
            ..k.clone()
        };
        let k = &renamed;
        self.reset_function_state();

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
        if let Some(d) = &mut self.debug {
            d.begin_function(&k.name, &k.name, k.span.line, k.span.col, &k.params, None);
        }

        for p in &k.params {
            self.emit_param_slot(p);
        }

        writeln!(
            &mut self.output,
            "  {} = alloca [8 x i8], align 8",
            Y_OOB_SINK
        )
        .unwrap();

        // A kernel whose whole body is the canonical matmul nest is replaced by
        // the packed AVX-512 kernel. The recogniser is strict and the scalar
        // lowering below is correct, so a near-miss costs speed, not an answer.
        let certificates_before = self.exact_gemm_certificates.len();
        let fast_start = self.output.len();
        if let Some(shape) = self.try_emit_gemm_kernel(k) {
            // Keep the original AST as an executable fallback. A recognised
            // shape does not establish that its three caller-owned buffers
            // are independent or that its span arithmetic cannot overflow.
            let fast_ir = self.output.split_off(fast_start);
            self.emit_alloca_for_block(&k.body);
            let scalar = self.fresh_label("gemm.scalar");
            let fast = self.fresh_label("gemm.fast");
            let done = self.fresh_label("gemm.return");
            self.emit_gemm_buffer_dispatch(&shape, &fast, &scalar, &done);
            writeln!(&mut self.output, "{fast}:").unwrap();
            self.output.push_str(&fast_ir);
            writeln!(&mut self.output, "  br label %{done}").unwrap();
            writeln!(&mut self.output, "{scalar}:").unwrap();
            self.block_terminated = false;
            self.emit_block_body(&k.body, "void");
            if !self.block_terminated {
                writeln!(&mut self.output, "  br label %{done}").unwrap();
            }
            writeln!(&mut self.output, "{done}:").unwrap();
            self.needs_gemm_module = true;
            // What runs is decided when the program runs, and `ydb verify`
            // has to say so: the substituted GEMM when the buffers are
            // disjoint, the body as written otherwise.
            if let Some(d) = self.debug.as_mut() {
                let exact = self.exact_gemm_certificates.len() > certificates_before;
                let (status, detail) = if exact {
                    (
                        crate::guarantees::Status::Proved,
                        "this kernel runs Y's exact vpdpwssd GEMM in place of its body when its \
three buffers do not overlap and their extents fit in 64 bits - checked when the program runs; \
otherwise the body runs as written. proofs/ExactGemmWhole.v proves the substituted kernel holds this nest's \
dot products exactly, for every shape - provided every operand lies within the @bounds on \
its operand `let`, which nothing checks when the program runs. `Y --emit-llvm` writes the \
certificate that instantiates the proof for this nest. Everything below the LLVM IR - clang, \
the assembler, the processor - is trusted"
                            .to_string(),
                    )
                } else {
                    (
                        crate::guarantees::Status::Tested,
                        "this kernel runs Y's packed, threaded f32 GEMM in place of its body when its \
three buffers do not overlap and their extents fit in 64 bits - checked when the program runs; \
otherwise the body runs as written. The GEMM is NOT bit-identical to the nest as written: f32 addition is not \
associative, so a tiled reduction rounds differently. It is tested against the nest \
(tests/gemm_substitution_differential.rs), not proved; Y_NO_GEMM_RECOGNISER=1 compiles the \
nest as written"
                            .to_string(),
                    )
                };
                let end_line = crate::ast::last_line(&k.body).max(k.span.line);
                // The licence is granted from the operands' `@bounds`, which
                // nothing checks: the exactness claim rests on them. Every
                // trusted range in the kernel is named - a superset, which is
                // the safe direction for a list of assumptions.
                let rests_on = if exact {
                    d.trusted_bounds(&k.name, k.span.line, end_line)
                } else {
                    Vec::new()
                };
                d.add_fact(crate::guarantees::Fact {
                    item: k.name.clone(),
                    line: k.span.line,
                    col: k.span.col,
                    end_line,
                    kind: "gemm",
                    status,
                    what: format!("kernel {}", k.name),
                    detail,
                    rests_on,
                });
            }
            writeln!(&mut self.output, "  ; [Y CPU GEMM] {:?}", shape).unwrap();
            self.wln("  ret void");
            self.wln("}");
            self.wln("");
            return;
        }

        self.emit_alloca_for_block(&k.body);

        self.emit_block_body(&k.body, "void");
        if !self.block_terminated {
            // As in `emit_func`: the implicit return belongs to the last statement.
            let outer = match k.body.stmts.last() {
                Some(last) => self.dbg_enter(&last.span()),
                None => None,
            };
            self.wln("  ret void");
            self.dbg_leave(outer);
        }
        self.wln("}");
        self.wln("");
    }

    /// Select a packed GEMM only when the live buffer ranges are disjoint.
    /// The source permits aliases; LLVM's packed routines do not. Invalid
    /// strides or overflowing span/end arithmetic go to the original body.
    fn emit_gemm_buffer_dispatch(
        &mut self,
        shape: &crate::cpu_gemm::GemmShape,
        fast: &str,
        scalar: &str,
        done: &str,
    ) {
        let mut values = Vec::new();
        for name in [
            &shape.m, &shape.n, &shape.k, &shape.lda, &shape.ldb, &shape.ldc,
        ] {
            let ty = self
                .locals
                .get(name)
                .expect("recognised GEMM header")
                .clone();
            let value = self.emit_load(&format!("%{name}"), &ty);
            values.push(self.emit_coerce(&value, &ty, "i64"));
        }
        let m_empty = self.fresh_tmp();
        let n_empty = self.fresh_tmp();
        let empty = self.fresh_tmp();
        let check_k = self.fresh_label("gemm.alias_check_k");
        let ranges = self.fresh_label("gemm.alias_ranges");
        let k_empty = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {m_empty} = icmp sle i64 {}, 0",
            values[0]
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  {n_empty} = icmp sle i64 {}, 0",
            values[1]
        )
        .unwrap();
        writeln!(&mut self.output, "  {empty} = or i1 {m_empty}, {n_empty}").unwrap();
        writeln!(
            &mut self.output,
            "  br i1 {empty}, label %{done}, label %{check_k}"
        )
        .unwrap();
        writeln!(&mut self.output, "{check_k}:").unwrap();
        writeln!(
            &mut self.output,
            "  {k_empty} = icmp sle i64 {}, 0",
            values[2]
        )
        .unwrap();
        // An empty contraction still writes C. Its scalar body never reads
        // A/B, so no input pointer/range condition is relevant to this case.
        writeln!(
            &mut self.output,
            "  br i1 {k_empty}, label %{scalar}, label %{ranges}"
        )
        .unwrap();
        writeln!(&mut self.output, "{ranges}:").unwrap();

        let mut requirements = Vec::new();
        let mut buffers = Vec::new();
        for (name, rows, cols, stride) in [
            (&shape.a, &values[0], &values[2], &values[3]),
            (&shape.b, &values[2], &values[1], &values[4]),
            (&shape.c, &values[0], &values[1], &values[5]),
        ] {
            let valid_stride = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {valid_stride} = icmp sge i64 {stride}, {cols}"
            )
            .unwrap();
            requirements.push(valid_stride);
            let prior_rows = self.fresh_tmp();
            writeln!(&mut self.output, "  {prior_rows} = sub i64 {rows}, 1").unwrap();
            let prefix =
                self.emit_gemm_checked_uint("umul", &prior_rows, stride, &mut requirements);
            let elements = self.emit_gemm_checked_uint("uadd", &prefix, cols, &mut requirements);
            let elem_ty = self
                .mem_elem_types
                .get(name)
                .expect("recognised GEMM buffer");
            let bytes_per_element = match elem_ty.as_str() {
                "i16" => "2",
                "i64" => "8",
                "float" => "4",
                _ => unreachable!("unrecognised GEMM element type"),
            };
            let bytes = self.emit_gemm_checked_uint(
                "umul",
                &elements,
                bytes_per_element,
                &mut requirements,
            );
            let signed_size = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {signed_size} = icmp ule i64 {bytes}, 9223372036854775807"
            )
            .unwrap();
            requirements.push(signed_size);
            let pointer = self.emit_load(&format!("%{name}"), "ptr");
            let address = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {address} = ptrtoint ptr {pointer} to i64"
            )
            .unwrap();
            let end = self.emit_gemm_checked_uint("uadd", &address, &bytes, &mut requirements);
            buffers.push((address, end));
        }
        for (left, right) in [(0, 1), (0, 2), (1, 2)] {
            let before = self.fresh_tmp();
            let after = self.fresh_tmp();
            let separate = self.fresh_tmp();
            writeln!(
                &mut self.output,
                "  {before} = icmp ule i64 {}, {}",
                buffers[left].1, buffers[right].0
            )
            .unwrap();
            writeln!(
                &mut self.output,
                "  {after} = icmp ule i64 {}, {}",
                buffers[right].1, buffers[left].0
            )
            .unwrap();
            writeln!(&mut self.output, "  {separate} = or i1 {before}, {after}").unwrap();
            requirements.push(separate);
        }
        let mut safe = "true".to_string();
        for requirement in requirements {
            let both = self.fresh_tmp();
            writeln!(&mut self.output, "  {both} = and i1 {safe}, {requirement}").unwrap();
            safe = both;
        }
        writeln!(
            &mut self.output,
            "  br i1 {safe}, label %{fast}, label %{scalar}"
        )
        .unwrap();
    }

    fn emit_gemm_checked_uint(
        &mut self,
        operation: &str,
        lhs: &str,
        rhs: &str,
        requirements: &mut Vec<String>,
    ) -> String {
        let pair = self.fresh_tmp();
        let value = self.fresh_tmp();
        let overflow = self.fresh_tmp();
        let fits = self.fresh_tmp();
        writeln!(&mut self.output, "  {pair} = call {{ i64, i1 }} @llvm.{operation}.with.overflow.i64(i64 {lhs}, i64 {rhs})").unwrap();
        writeln!(
            &mut self.output,
            "  {value} = extractvalue {{ i64, i1 }} {pair}, 0"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  {overflow} = extractvalue {{ i64, i1 }} {pair}, 1"
        )
        .unwrap();
        writeln!(&mut self.output, "  {fits} = xor i1 {overflow}, true").unwrap();
        requirements.push(fits);
        value
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
            &shape.m, &shape.n, &shape.k, &shape.lda, &shape.ldb, &shape.ldc,
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
                crate::cpu_gemm::ExactGemmPlan::Vnni {
                    scheme,
                    operand_magnitude,
                } => {
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
            &shape.m, &shape.n, &shape.k, &shape.lda, &shape.ldb, &shape.ldc,
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

        // The source has signed ascending loops. Nonpositive M/N perform no
        // stores; nonpositive K with positive M/N writes the initial +0.0
        // accumulator without reading A/B. Keep those cases out of the packed
        // kernel, whose scheduling and allocation arithmetic needs positive
        // extents. In particular, a negative N is not a memset byte count.
        let m_empty = self.fresh_tmp();
        let n_empty = self.fresh_tmp();
        let empty = self.fresh_tmp();
        let k_empty = self.fresh_tmp();
        let check_k = self.fresh_label("gemm.check_k");
        let zero = self.fresh_label("gemm.zero");
        let zero_cond = self.fresh_label("gemm.zero_cond");
        let zero_body = self.fresh_label("gemm.zero_body");
        let compute = self.fresh_label("gemm.compute");
        let done = self.fresh_label("gemm.done");
        writeln!(&mut self.output, "  {m_empty} = icmp sle i64 {}, 0", ext[0]).unwrap();
        writeln!(&mut self.output, "  {n_empty} = icmp sle i64 {}, 0", ext[1]).unwrap();
        writeln!(&mut self.output, "  {empty} = or i1 {m_empty}, {n_empty}").unwrap();
        writeln!(
            &mut self.output,
            "  br i1 {empty}, label %{done}, label %{check_k}"
        )
        .unwrap();
        writeln!(&mut self.output, "{check_k}:").unwrap();
        writeln!(&mut self.output, "  {k_empty} = icmp sle i64 {}, 0", ext[2]).unwrap();
        writeln!(
            &mut self.output,
            "  br i1 {k_empty}, label %{zero}, label %{compute}"
        )
        .unwrap();

        writeln!(&mut self.output, "{zero}:").unwrap();
        let row_bytes = self.fresh_tmp();
        let row = self.fresh_tmp();
        let next_row = self.fresh_tmp();
        let more = self.fresh_tmp();
        let offset = self.fresh_tmp();
        let row_ptr = self.fresh_tmp();
        writeln!(&mut self.output, "  {row_bytes} = mul i64 {}, 4", ext[1]).unwrap();
        writeln!(&mut self.output, "  br label %{zero_cond}").unwrap();
        writeln!(&mut self.output, "{zero_cond}:").unwrap();
        writeln!(
            &mut self.output,
            "  {row} = phi i64 [ 0, %{zero} ], [ {next_row}, %{zero_body} ]"
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  {more} = icmp slt i64 {row}, {}",
            ext[0]
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  br i1 {more}, label %{zero_body}, label %{done}"
        )
        .unwrap();
        writeln!(&mut self.output, "{zero_body}:").unwrap();
        writeln!(&mut self.output, "  {offset} = mul i64 {row}, {}", ext[5]).unwrap();
        writeln!(
            &mut self.output,
            "  {row_ptr} = getelementptr float, ptr {}, i64 {offset}",
            ptrs[2]
        )
        .unwrap();
        writeln!(
            &mut self.output,
            "  call void @llvm.memset.p0.i64(ptr {row_ptr}, i8 0, i64 {row_bytes}, i1 false)"
        )
        .unwrap();
        writeln!(&mut self.output, "  {next_row} = add i64 {row}, 1").unwrap();
        writeln!(&mut self.output, "  br label %{zero_cond}").unwrap();

        writeln!(&mut self.output, "{compute}:").unwrap();
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
        writeln!(&mut self.output, "  br label %{done}").unwrap();
        writeln!(&mut self.output, "{done}:").unwrap();
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

    /// Every statement goes through here, which is what attributes its code to
    /// its own source line under `-g`: the position is set on entry and put
    /// back on exit, so what a compound statement emits after its body (a
    /// loop's increment and back edge, an `if`'s jump to its merge block)
    /// belongs to the compound statement and not to its last inner one.
    fn emit_stmt(&mut self, stmt: &Stmt, ret_type: &str) {
        let outer = self.dbg_enter(&stmt.span());
        self.emit_stmt_inner(stmt, ret_type);
        // The binding exists once its initialiser has been stored, not before:
        // the `let`'s own code runs in the enclosing scope.
        if let Stmt::Let { name, span, .. } = stmt {
            self.dbg_bind(name, span, true);
        }
        self.dbg_leave(outer);
    }

    /// The jump out of the end of a block - an `if` branch to its merge block,
    /// a `while` body back to its condition. Under `-g` it belongs to the
    /// block's last statement: attributed to the `if` or `while` itself, where
    /// `emit_stmt` would leave it, `next` would stop on that line a second time
    /// on the way out of a `then` branch and on every iteration of a loop.
    /// (A `for` loop's increment and back edge DO belong to the `for` line:
    /// they are its header's code.)
    fn emit_branch_out(&mut self, branch: &Block, label: &str) {
        let outer = match branch.stmts.last() {
            Some(last) => self.dbg_enter(&last.span()),
            None => None,
        };
        writeln!(&mut self.output, "  br label %{}", label).unwrap();
        self.dbg_leave(outer);
    }

    fn emit_stmt_inner(&mut self, stmt: &Stmt, ret_type: &str) {
        match stmt {
            Stmt::Let { name, init, .. } if self.zero_drift.contains_key(name) => {
                let (repr, integer_domain) = self.zero_drift[name];
                let fixed = match init {
                    Some(e) => self.emit_drift_term(e, repr, integer_domain),
                    None => "0".to_string(),
                };
                self.emit_store(&fixed, &format!("%{}", name), repr.llvm_type());
            }
            Stmt::Let { name, init: Some(init_expr), .. } if self.local_q_format(name).is_some() => {
                let fmt = self.local_q_format(name).unwrap();
                let raw = self.q_value(init_expr, fmt);
                self.emit_store(&raw, &format!("%{}", name), &fmt.llvm());
            }
            // A `@cache_policy` on this `let` never reaches here: `emit_program`
            // refuses the whole program first (see the comment there).
            Stmt::Let { name, init, .. } => {
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
                        let coerced = self.emit_coerce_from(&val, &val_ty, &dst_ty, src_unsigned);

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

                            let is_aggregate_type =
                                val_ty.starts_with('%') || val_ty.starts_with('[');
                            let is_registered_type =
                                self.structs.contains_key(dst_ty.trim_start_matches('%'))
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
            Stmt::Assign {
                target,
                value,
                span,
            } if matches!(target, Expr::Ident(n, _) if self.zero_drift.contains_key(n))
                && Self::drift_running_sum(target, value).is_some() =>
            {
                let name = match target {
                    Expr::Ident(n, _) => n.clone(),
                    _ => unreachable!(),
                };
                let (op, rhs) = Self::drift_running_sum(target, value).unwrap();
                let (repr, integer_domain) = self.zero_drift[&name];
                let rhs_fixed = self.emit_drift_term(rhs, repr, integer_domain);
                let ity = repr.llvm_type();
                let addr = format!("%{}", name);
                let loaded = self.emit_load(&addr, ity);
                let result = self.fresh_tmp();
                let instr = if matches!(op, BinaryOp::Sub) {
                    "sub"
                } else {
                    "add"
                };
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
            Stmt::Assign { target, span, .. } if matches!(target, Expr::Ident(n, _) if self.zero_drift.contains_key(n)) =>
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
            Stmt::Assign { target, value, .. } if self.q_format(target).is_some() => {
                let fmt = self.q_format(target).unwrap();
                let addr = self.emit_lvalue(target);
                let raw = self.q_value(value, fmt);
                self.emit_store(&raw, &addr, &fmt.llvm());
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
                        let is_registered_type =
                            self.structs.contains_key(dst_ty.trim_start_matches('%'))
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
            Stmt::Return(Some(e), _) if self.current_ret_q.is_some() => {
                let fmt = self.current_ret_q.unwrap();
                let raw = self.q_value(e, fmt);
                writeln!(&mut self.output, "  ret {} {}", fmt.llvm(), raw).unwrap();
                self.block_terminated = true;
            }
            Stmt::Return(expr, _) => {
                if let Some(e) = expr {
                    let val = self.emit_expr(e, None, None);
                    let val_ty = self.infer_type(e);
                    let src_unsigned = self.expr_is_unsigned(e);
                    let coerced = self.emit_coerce_from(&val, &val_ty, ret_type, src_unsigned);
                    writeln!(&mut self.output, "  ret {} {}", ret_type, coerced).unwrap();
                } else {
                    if ret_type == "void" {
                        self.wln("  ret void");
                    } else if self.aot_entry_status && ret_type == "i32" {
                        self.wln("  ret i32 0");
                    } else {
                        self.emit_errors.push("[LLVM host backend] a value-returning function requires a return value".into());
                        writeln!(&mut self.output, "  ret {} zeroinitializer", ret_type).unwrap();
                    }
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
                let scope = self.dbg_scope_enter(&then_block.span);
                self.emit_block_body(then_block, ret_type);
                let then_terminated = self.block_terminated;
                if !then_terminated {
                    self.emit_branch_out(then_block, &merge_lbl);
                }
                self.dbg_scope_leave(scope);

                // Else block
                if let Some(eb) = else_block {
                    writeln!(&mut self.output, "{}:", else_lbl).unwrap();
                    self.block_terminated = false;
                    let scope = self.dbg_scope_enter(&eb.span);
                    self.emit_block_body(eb, ret_type);
                    let else_terminated = self.block_terminated;
                    if !else_terminated {
                        self.emit_branch_out(eb, &merge_lbl);
                    }
                    self.dbg_scope_leave(scope);
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
                condition,
                body,
                is_uniform_branch,
                ..
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
                let scope = self.dbg_scope_enter(&body.span);
                self.emit_block_body(body, ret_type);
                self.loop_exit_stack.pop();
                if !self.block_terminated {
                    self.emit_branch_out(body, &cond_lbl);
                }
                self.dbg_scope_leave(scope);

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
                span,
                ..
            } => {
                // Source induction variables are I32. Safe-loop verification
                // proves each header value fits that range before lowering;
                // a wider or narrower operand still needs an explicit LLVM
                // conversion rather than using its register at the wrong type.
                let s_value = self.emit_expr(start, None, None);
                let s_ty = self.infer_type(start);
                let s_unsigned = self.expr_is_unsigned(start);
                let s = self.emit_coerce_from(&s_value, &s_ty, "i32", s_unsigned);
                let e_value = self.emit_expr(end, None, None);
                let e_ty = self.infer_type(end);
                let e_unsigned = self.expr_is_unsigned(end);
                let e = self.emit_coerce_from(&e_value, &e_ty, "i32", e_unsigned);
                // `start`, `end` and `step` are each evaluated ONCE, before
                // the first iteration, in that order - as the PTX backend and
                // `--emit-cpu` do. The step used to be evaluated at every
                // increment, after the body, so a body that changed it
                // (`for i in 0..20 step k { k = k + 1; }`) ran 5 iterations
                // here and 20 on the GPU.
                let step_val = if let Some(st) = step {
                    let value = self.emit_expr(st, None, None);
                    let ty = self.infer_type(st);
                    let unsigned = self.expr_is_unsigned(st);
                    self.emit_coerce_from(&value, &ty, "i32", unsigned)
                } else {
                    "1".into()
                };
                let cond_lbl = self.fresh_label("for.cond");
                let body_lbl = self.fresh_label("for.body");
                let end_lbl = self.fresh_label("for.end");

                if let Some(t) = tile {
                    writeln!(
                        &mut self.output,
                        "  ; [Y TILE OPTIMIZATION] Tiled loop dimensions: M={:?}, N={:?}, K={:?}",
                        t.block_m, t.block_n, t.block_k
                    )
                    .unwrap();
                }

                // `@prefetch_stride` never reaches here: the type checker
                // refuses it, because no backend lowers it. This arm used to
                // write it into the module as a COMMENT ("solver-guided cache
                // warming") and emit no prefetch.

                let metadata = if *is_uniform_branch {
                    ", !uniform_branch !0 ; Maps to BRANCH_UNIFORM_CYCLES scheduling baseline"
                } else {
                    ""
                };

                // alloca is in entry
                self.emit_store(&s, &format!("%{}", loop_var), "i32");
                // The loop variable exists from its first value on - through
                // the condition, the body and the increment, which are all the
                // loop's - and not after the loop.
                let loop_scope = self.dbg_scope_enter(span);
                self.dbg_bind(loop_var, span, false);
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
                self.emit_scoped_block(body, ret_type);
                self.loop_exit_stack.pop();

                // Increment, by the step evaluated before the loop.
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
                self.dbg_scope_leave(loop_scope);

                writeln!(&mut self.output, "{}:", end_lbl).unwrap();
                self.block_terminated = false;
            }
            Stmt::CompoundAssign {
                target,
                op,
                value,
                span,
            } if matches!(target, Expr::Ident(n, _) if self.zero_drift.contains_key(n)) => {
                let name = match target {
                    Expr::Ident(n, _) => n.clone(),
                    _ => unreachable!(),
                };
                let (repr, integer_domain) = self.zero_drift[&name];
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
                let rhs_fixed = self.emit_drift_term(value, repr, integer_domain);
                let ity = repr.llvm_type();
                let addr = format!("%{}", name);
                let loaded = self.emit_load(&addr, ity);
                let result = self.fresh_tmp();
                let instr = if matches!(op, BinaryOp::Sub) {
                    "sub"
                } else {
                    "add"
                };
                writeln!(
                    &mut self.output,
                    "  {} = {} {} {}, {}",
                    result, instr, ity, loaded, rhs_fixed
                )
                .unwrap();
                self.emit_store(&result, &addr, ity);
            }
            Stmt::CompoundAssign { target, op, value, .. } if self.q_format(target).is_some() => {
                let fmt = self.q_format(target).unwrap();
                let addr = self.emit_lvalue(target);
                let current = self.emit_load(&addr, &fmt.llvm());
                let rhs = self.q_value(value, fmt);
                let result = self.q_operation(op, &current, &rhs, fmt);
                self.emit_store(&result, &addr, &fmt.llvm());
            }
            Stmt::CompoundAssign {
                target, op, value, ..
            } => {
                let addr = self.emit_lvalue(target);
                let rhs = self.emit_expr(value, None, None);
                let ty = self.infer_type(target);
                let loaded = self.emit_load(&addr, &ty);
                let (loaded, rhs, op_ty) =
                    self.promote_binary_values(op, target, value, &loaded, &rhs);
                let unsigned = self.binary_is_unsigned(op, target, value);
                let result = self.fresh_tmp();
                let op_str = self.binop_to_llvm(op, &op_ty, unsigned);
                writeln!(
                    &mut self.output,
                    "  {} = {} {} {}, {}",
                    result, op_str, op_ty, loaded, rhs
                )
                .unwrap();
                let result = self.emit_coerce_from(&result, &op_ty, &ty, unsigned);
                self.emit_store(&result, &addr, &ty);
            }
            Stmt::Chisel(block, span) => {
                if self.in_ptx_emit {
                    // This arm emitted the `chisel` lines as inline asm with an
                    // EMPTY constraint string, on the reading that a module
                    // retargeted to NVPTX wants different constraints from x86.
                    // `emit_prelude` writes `Self::host_triple()` and there is
                    // no path in this backend that emits an `nvptx` triple, so
                    // the reading never applies: what came out was PTX text
                    // inside an `x86_64-unknown-linux-gnu` module. Measured -
                    // the compiler printed "Compilation Successful!", exited 0,
                    // and the `clang` line it told the user to run answered
                    // `<inline asm>:1:10: error: invalid register name`.
                    //
                    // Both branches failed identically there, so `@ptx_emit`'s
                    // one live consumer could not change an outcome. Refusing
                    // by name is what makes that status honest: `--emit-ptx` is
                    // where PTX `chisel` is lowered, and it resolves `%name`
                    // against the variables in scope (`resolve_chisel_registers`).
                    self.emit_errors.push(format!(
                        "[LLVM] a `chisel` block inside a `@ptx_emit` function (line {}, \
                         col {}) would put PTX instructions into a `{}` module, which \
                         `clang` rejects with `invalid register name`. This backend emits \
                         host code only. Use `--emit-ptx` with a `kernel`, or drop \
                         `@ptx_emit` and write host assembly.",
                        span.line,
                        span.col,
                        Self::host_triple().0
                    ));
                } else {
                    self.wln("  ; --- CHISEL INLINE ASM ---");
                    let scope = self.dbg_scope_enter(&block.span);
                    for stmt in &block.stmts {
                        if let Stmt::Expr(Expr::StringLit(s, _)) = stmt {
                            self.wln(&format!("  call void asm sideeffect \"{}\", \"~{{memory}},~{{dirflag}},~{{fpsr}},~{{flags}}\"()", s));
                        } else {
                            self.emit_stmt(stmt, ret_type);
                        }
                    }
                    self.dbg_scope_leave(scope);
                }
            }
            Stmt::Match {
                scrutinee, arms, ..
            } => {
                if let Some(fmt) = self.q_format(scrutinee) {
                    self.emit_errors.push(format!(
                        "[LLVM host backend] `match` on a {} value: its patterns are not \
                         lowered in fixed point; compare with `if` instead",
                        fmt.name()
                    ));
                }
                let scrut_val = self.emit_expr(scrutinee, None, None);
                let scrut_ty = self.infer_type(scrutinee);
                let scrut_ast = self.infer_ast_type(scrutinee);
                let enum_name = if scrut_ty.starts_with('%') { scrut_ty.trim_start_matches('%') } else { scrut_ast.as_str() };
                let scrut_tag = if self.enums.get(enum_name) == Some(&true) {
                    let tag = self.fresh_tmp();
                    writeln!(&mut self.output, "  {tag} = extractvalue {scrut_ty} {scrut_val}, 0").unwrap();
                    tag
                } else { scrut_val.clone() };
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
                    let mut payload_layout = None;
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
                            if self.enums.contains_key(enum_name) {
                                self.emit_errors.push("[LLVM host backend] enum matches require variant patterns, a binding or `_`".into());
                                writeln!(&mut self.output, "  br label %{next_test}").unwrap();
                            } else {
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
                        }
                        MatchPattern::Ident(_, _) => {
                            writeln!(&mut self.output, "  br label %{body_lbl}").unwrap();
                        }
                        MatchPattern::EnumVariant { path, variant, bindings, .. } => {
                            let namespace = if path.is_empty() { enum_name } else { path.as_str() };
                            let key = format!("{namespace}_{variant}");
                            if let Some(layout) = self.enum_variant_layouts.get(&key).cloned().filter(|layout|
                                layout.enum_name == enum_name && bindings.len() == layout.fields.len()) {
                                let tag = self.enum_variants[&key];
                                let cmp = self.fresh_tmp();
                                writeln!(&mut self.output, "  {cmp} = icmp eq i32 {scrut_tag}, {tag}\n  br i1 {cmp}, label %{body_lbl}, label %{next_test}").unwrap();
                                payload_layout = Some(layout);
                            } else {
                                self.emit_errors.push(format!("[LLVM host backend] invalid enum match pattern `{namespace}::{variant}` for `{enum_name}` or incorrect binding count"));
                                writeln!(&mut self.output, "  br label %{next_test}").unwrap();
                            }
                        }
                    }

                    writeln!(&mut self.output, "{}:", body_lbl).unwrap();
                    self.block_terminated = false;
                    let saved_locals = self.locals.clone();
                    let saved_ast_types = self.locals_ast_type.clone();
                    let saved_pointees = self.pointee_types.clone();
                    match &arm.pattern {
                        MatchPattern::Ident(name, _) => {
                            self.locals.insert(name.clone(), scrut_ty.clone());
                            self.locals_ast_type.insert(name.clone(), scrut_ast.clone());
                            if scrut_ty.starts_with('%') { self.pointee_types.insert(name.clone(), scrut_ty.clone()); }
                            writeln!(&mut self.output, "  %{name} = alloca {scrut_ty}\n  store {scrut_ty} {scrut_val}, ptr %{name}").unwrap();
                        }
                        MatchPattern::EnumVariant { bindings, .. } => {
                            if let Some(layout) = payload_layout {
                                let storage = self.fresh_tmp();
                                writeln!(&mut self.output, "  {storage} = alloca {scrut_ty}, align 8\n  store {scrut_ty} {scrut_val}, ptr {storage}, align 8").unwrap();
                                for (index, (binding, (field_ty, ast_ty))) in bindings.iter().zip(&layout.fields).enumerate() {
                                    let pointer = self.fresh_tmp();
                                    writeln!(&mut self.output, "  {pointer} = getelementptr {scrut_ty}, ptr {storage}, i32 0, i32 1, i32 {index}").unwrap();
                                    let value = self.emit_load(&pointer, field_ty);
                                    self.locals.insert(binding.clone(), field_ty.clone());
                                    self.locals_ast_type.insert(binding.clone(), ast_ty.clone());
                                    writeln!(&mut self.output, "  %{binding} = alloca {field_ty}\n  store {field_ty} {value}, ptr %{binding}").unwrap();
                                }
                            }
                        }
                        _ => {}
                    }
                    let outer = self.dbg_enter(&arm.span);
                    self.emit_expr(&arm.body, None, None);
                    self.dbg_leave(outer);
                    self.locals = saved_locals;
                    self.locals_ast_type = saved_ast_types;
                    self.pointee_types = saved_pointees;
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
                self.emit_scoped_block(block, ret_type);
            }
            Stmt::GhostBlock(block, _) => {
                self.wln("  ; --- @ghost speculative block ---");
                self.emit_scoped_block(block, ret_type);
            }
            Stmt::HintBlock { body, .. } => {
                self.wln("  ; --- @hint unconstrained block ---");
                self.emit_scoped_block(body, ret_type);
            }
            Stmt::ClockDomainBlock { body, .. } => {
                self.wln("  ; --- @clock_domain block ---");
                self.emit_scoped_block(body, ret_type);
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
        match self.enum_payload_field(expr) {
            Ok(Some((owner, layout, index))) => {
                let owner_ast = self.infer_ast_type(&owner);
                let pointer = if owner_ast.starts_with('&') {
                    self.emit_expr(&owner, None, None)
                } else {
                    self.emit_lvalue(&owner)
                };
                let result = self.fresh_tmp();
                let ty = self.enum_value_type(&layout);
                writeln!(&mut self.output, "  {result} = getelementptr {ty}, ptr {pointer}, i32 0, i32 1, i32 {index}").unwrap();
                return result;
            }
            Err(message) => {
                self.emit_errors.push(message);
                return "null".into();
            }
            Ok(None) => {}
        }
        match expr {
            Expr::Ident(name, _) => format!("%{}", name),
            Expr::MemberAccess { base, member, .. } => {
                let (base_val, base_ty) = if let Expr::UnaryOp {
                    op: UnaryOp::Deref,
                    operand: inner,
                    ..
                } = &**base
                {
                    (
                        self.emit_expr(inner, None, None),
                        self.infer_struct_type(inner),
                    )
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
                        self.emit_errors.push("[LLVM host backend] enum payload fields require a declared variant overlay (`value.data.Variant._N`)".into());
                        return "null".into();
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

                let is_safe = crate::type_checker::SAFE_INDICES
                    .with(|set| set.borrow().contains(&(span.line, span.col)));
                let array_size = crate::type_checker::INDEX_ARRAY_SIZES
                    .with(|map| map.borrow().get(&(span.line, span.col)).cloned());

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
                        )
                        .unwrap();
                        writeln!(
                            &mut self.output,
                            "  br i1 {}, label %{}, label %{}",
                            cond, fail_lbl, ok_lbl
                        )
                        .unwrap();

                        // Fail block
                        writeln!(&mut self.output, "{}:", fail_lbl).unwrap();
                        let idx_i64 = if cmp_ty == "i32" {
                            let tmp = self.fresh_tmp();
                            writeln!(&mut self.output, "  {} = sext i32 {} to i64", tmp, idx_val)
                                .unwrap();
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
                    match self.pointer_elem_type(base) {
                        Ok(t) => t,
                        Err(why) => {
                            self.emit_errors.push(format!(
                                "[LLVM host backend] {}. Indexing used to step and store 8 bytes per \
                                 element whatever the element was; it is refused rather than guessed.",
                                why
                            ));
                            "i8".to_string()
                        }
                    }
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

    /// Recognize `(x << a) | (x >> b)` and its right-rotate form only when
    /// both shifts have the same promoted width and valid complementary counts.
    /// Funnel shifts mask their count, while ordinary shifts produce poison for
    /// an out-of-range count, so zero/width and unproved dynamic counts stay as
    /// ordinary shifts. Only local identifiers can be shared: calls, pointer
    /// reads, and indexed/field expressions retain their original evaluations.
    fn try_emit_rotate(&mut self, left: &Expr, right: &Expr) -> Option<String> {
        if !self.recognize_rotates {
            return None;
        }
        let Expr::BinaryOp {
            left: l_value,
            op: l_op,
            right: l_count,
            ..
        } = left
        else {
            return None;
        };
        let Expr::BinaryOp {
            left: r_value,
            op: r_op,
            right: r_count,
            ..
        } = right
        else {
            return None;
        };
        let intrinsic = match (l_op, r_op) {
            (BinaryOp::Shl, BinaryOp::Shr) => "fshl",
            (BinaryOp::Shr, BinaryOp::Shl) => "fshr",
            _ => return None,
        };
        let (Expr::Ident(l_name, _), Expr::Ident(r_name, _)) = (&**l_value, &**r_value) else {
            return None;
        };
        if l_name != r_name || !self.locals.contains_key(l_name) || !self.expr_is_unsigned(l_value)
        {
            return None;
        }

        // Only small, pure integer constants are folded. Every intermediate
        // fits i32, matching literal arithmetic without overflow assumptions.
        fn count(expr: &Expr) -> Option<i64> {
            let value = match expr {
                Expr::IntLit(value, _) => *value,
                Expr::BinaryOp {
                    left,
                    op: BinaryOp::Add,
                    right,
                    ..
                } => count(left)?.checked_add(count(right)?)?,
                Expr::BinaryOp {
                    left,
                    op: BinaryOp::Sub,
                    right,
                    ..
                } => count(left)?.checked_sub(count(right)?)?,
                _ => return None,
            };
            (0..=64).contains(&value).then_some(value)
        }
        let ty = self.infer_type(left);
        if ty != self.infer_type(right) {
            return None;
        }
        let width = match ty.as_str() {
            "i8" => 8,
            "i16" => 16,
            "i32" => 32,
            "i64" => 64,
            _ => return None,
        };
        let l_count = count(l_count)?;
        let r_count = count(r_count)?;
        if l_count <= 0
            || r_count <= 0
            || l_count >= width
            || r_count >= width
            || l_count + r_count != width
        {
            return None;
        }

        let value = self.emit_expr(l_value, None, None);
        let source_ty = self.infer_type(l_value);
        let value = self.emit_coerce_from(&value, &source_ty, &ty, true);
        let result = self.fresh_tmp();
        writeln!(&mut self.output,
            "  {result} = call {ty} @llvm.{intrinsic}.{ty}({ty} {value}, {ty} {value}, {ty} {l_count})"
        ).unwrap();
        Some(result)
    }

    /// Evaluate the RHS only when the left boolean cannot decide the result.
    /// The dedicated incoming blocks make each PHI predecessor explicit even
    /// when nested logical expressions or bounds checks create RHS blocks.
    fn emit_short_circuit(&mut self, op: &BinaryOp, left: &Expr, right: &Expr) -> String {
        let lhs = self.emit_expr(left, None, None);
        let lhs_ty = self.infer_type(left);
        let lhs = self.emit_coerce_from(&lhs, &lhs_ty, "i1", false);
        let rhs_label = self.fresh_label("logic.rhs");
        let short_label = self.fresh_label("logic.short");
        let rhs_done_label = self.fresh_label("logic.rhs_done");
        let merge_label = self.fresh_label("logic.merge");
        let is_and = matches!(op, BinaryOp::And);
        let (if_true, if_false) = if is_and {
            (&rhs_label, &short_label)
        } else {
            (&short_label, &rhs_label)
        };
        writeln!(
            &mut self.output,
            "  br i1 {}, label %{}, label %{}",
            lhs, if_true, if_false
        )
        .unwrap();

        writeln!(&mut self.output, "{}:", short_label).unwrap();
        writeln!(&mut self.output, "  br label %{}", merge_label).unwrap();

        writeln!(&mut self.output, "{}:", rhs_label).unwrap();
        self.block_terminated = false;
        let rhs = self.emit_expr(right, None, None);
        let rhs_ty = self.infer_type(right);
        let rhs = self.emit_coerce_from(&rhs, &rhs_ty, "i1", false);
        writeln!(&mut self.output, "  br label %{}", rhs_done_label).unwrap();
        writeln!(&mut self.output, "{}:", rhs_done_label).unwrap();
        writeln!(&mut self.output, "  br label %{}", merge_label).unwrap();

        writeln!(&mut self.output, "{}:", merge_label).unwrap();
        self.block_terminated = false;
        let result = self.fresh_tmp();
        writeln!(
            &mut self.output,
            "  {} = phi i1 [ {}, %{} ], [ {}, %{} ]",
            result,
            if is_and { "false" } else { "true" },
            short_label,
            rhs,
            rhs_done_label
        )
        .unwrap();
        result
    }

    fn emit_expr(
        &mut self,
        expr: &Expr,
        target: Option<String>,
        expected_ty: Option<String>,
    ) -> String {
        if let Some(value) = self.emit_q_expr(expr) {
            return value;
        }
        match expr {
            Expr::IntLit(val, _) => format!("{}", val),
            // LLVM's hexadecimal form preserves every parsed IEEE-754 bit.
            // Decimal output with six places rounded 1.0000001 to 1.000000.
            Expr::FloatLit(val, _) => format!("0x{:016X}", val.to_bits()),
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

                // Integer observations stay exact too. Only a source-level
                // float needs to be decoded from its fixed-point storage.
                if let Some((repr, integer_domain)) = self.zero_drift.get(name).copied() {
                    let raw = self.emit_load(&format!("%{}", name), repr.llvm_type());
                    if integer_domain {
                        return raw;
                    }
                    return self.emit_from_fixed(&raw, repr);
                }
                let ty = self
                    .locals
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| "i32".into());
                // An array evaluates to its ADDRESS, the way an array-typed
                // struct field already does in the `MemberAccess` arm below:
                // indexing, `let`, assignment and argument passing all want
                // the storage, and every consumer that needs the aggregate
                // value (`memcpy`, `insertvalue`) takes it from there.
                if ty.starts_with('[') {
                    return format!("%{}", name);
                }
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
                if matches!(op, BinaryOp::And | BinaryOp::Or) {
                    return self.emit_short_circuit(op, left, right);
                }
                if matches!(op, BinaryOp::BitOr) {
                    if let Some(result) = self.try_emit_rotate(left, right) {
                        return result;
                    }
                }
                let l = self.emit_expr(left, None, None);
                let r = self.emit_expr(right, None, None);
                let (l, r, ty) = self.promote_binary_values(op, left, right, &l, &r);
                let unsigned = self.binary_is_unsigned(op, left, right);
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

                let instr = self.binop_to_llvm(op, &ty, unsigned);
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

                if let Some(value) = self.try_emit_runtime_query(&func_name, args) {
                    return value;
                }
                if let Some(value) = self.try_emit_runtime_append(&func_name, args) {
                    return value;
                }
                if let Some(value) = self.try_emit_runtime_bulk_append(&func_name, args) {
                    return value;
                }

                // Block-pointer intrinsics lower to native address arithmetic.
                // Routing them through the generic call path instead declared
                // them `i32 (...)`, which truncated the 64-bit base pointer and
                // then `sitofp`'d a float bit pattern — wrong answers, not just
                // slow ones. It also planted an opaque call in the innermost
                // loop, which blocks every loop and vector transform LLVM has.
                if let Some(v) = self.try_emit_block_ptr_intrinsic(&func_name, args) {
                    return v;
                }

                if let Some(value) = self.emit_enum_constructor(&func_name, args) {
                    return value;
                }
                if !self.fn_llvm_params.contains_key(&func_name) {
                    if let Some(fmt) = args.iter().find_map(|a| self.q_format(a)) {
                        self.emit_errors.push(format!(
                            "[LLVM host backend] `{}` would receive a {} value as its raw \
                             scaled integer; only a Y function with a {} parameter takes one",
                            func_name,
                            fmt.name(),
                            fmt.name()
                        ));
                    }
                }
                self.called_functions.push(func_name.clone());

                if (func_name.starts_with("String_")
                    || func_name.starts_with("Vec_")
                    || func_name.starts_with("File_")
                    || func_name.starts_with("yfile_")
                    || func_name.starts_with("ystr_")
                    || func_name.starts_with("yvec_"))
                    && args.len() >= 1
                    && !self.fn_llvm_params.contains_key(&func_name)
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

                        // eq_cstr accepts registered handles and raw C text.
                        // Direct &String locals are known slots; ambiguous
                        // typed references use a registry resolver. Genuine
                        // &char C text never enters this path.
                        if i == 1
                            && matches!(func_name.as_str(), "String_eq_cstr" | "ystr_eq_cstr")
                            && self.host_runtime_call(&func_name)
                            && matches!(arg_ast.as_str(), "&String" | "&mut String")
                        {
                            if matches!(arg, Expr::UnaryOp { op: UnaryOp::Ref { .. }, operand, .. }
                                if matches!(&**operand, Expr::Ident(local, _) if self.locals_ast_type.get(local).map(String::as_str) == Some("String")))
                            {
                                arg_val = self.emit_load(&arg_val, "ptr");
                            } else if let Some(address) = self.native_string_handle_normalizer {
                                let normalized = self.fresh_tmp();
                                writeln!(&mut self.output, "  {normalized} = call ptr inttoptr (i64 {address} to ptr)(ptr {arg_val})").unwrap();
                                arg_val = normalized;
                            }
                        }

                        let param_ty = expected_params.get(i).map(|s| s.as_str()).unwrap_or("i32");

                        if arg_ast.starts_with('&') && arg_ast[1..] == *param_ty {
                            let tmp = self.fresh_tmp();
                            writeln!(&mut self.output, "  {} = load ptr, ptr {}", tmp, arg_val)
                                .unwrap();
                            arg_val = tmp;
                        }

                        // The definition's own signature decides: `emit_type` per parameter,
                        // the computation `define` used. The table below is only for names no
                        // definition declares (the runtime's `String_*`/`Vec_*`); it knew neither
                        // `U32` nor arrays, and its `%<name>` fallback named a type the module
                        // never defines.
                        let declared = self
                            .fn_llvm_params
                            .get(&func_name)
                            .and_then(|tys| tys.get(i))
                            .cloned()
                            .flatten();
                        let llvm_param_ty = match declared {
                            Some(t) => t,
                            None => match param_ty {
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
                            },
                        };

                        if !arg_ty.starts_with('%')
                            && !llvm_param_ty.starts_with('%')
                            && arg_ty != "ptr"
                            && llvm_param_ty != "ptr"
                        {
                            let u = self.expr_is_unsigned(arg);
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

                    if func_name.starts_with("Vec_get_")
                        && args.len() == 2
                        && !(func_name == "Vec_get_char" && self.host_runtime_call(&func_name))
                    {
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
                        "String_char_at" | "ystr_char_at" | "yvec_get_char" | "Vec_get_char" => {
                            "i8".into()
                        }
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

                    // A Y function's Q parameter takes the argument in its own
                    // format: a literal `1.5` is 98304 to a Q16.16, never `fptosi`.
                    let q_param = if self.fn_llvm_params.contains_key(&func_name) {
                        QFormat::parse(param_ty)
                    } else {
                        None
                    };
                    let mut arg_val = match q_param {
                        Some(fmt) => self.q_value(a, fmt),
                        None => self.emit_expr(a, None, None),
                    };
                    let arg_ty = match q_param {
                        Some(fmt) => fmt.llvm(),
                        None => self.infer_type(a),
                    };
                    let arg_ast = self.infer_ast_type(a);

                    if arg_ast.starts_with('&') && arg_ast[1..] == *param_ty {
                        let tmp = self.fresh_tmp();
                        writeln!(&mut self.output, "  {} = load ptr, ptr {}", tmp, arg_val)
                            .unwrap();
                        arg_val = tmp;
                    }

                    // The definition's own signature decides: `emit_type` per parameter,
                    // the computation `define` used. The table below is only for names no
                    // definition declares (the runtime's `String_*`/`Vec_*`); it knew neither
                    // `U32` nor arrays, and its `%<name>` fallback named a type the module
                    // never defines.
                    let declared = self
                        .fn_llvm_params
                        .get(&func_name)
                        .and_then(|tys| tys.get(i))
                        .cloned()
                        .flatten();
                    let llvm_param_ty = match declared {
                        Some(t) => t,
                        None => match param_ty {
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
                        },
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
                            "  {} = load {}, ptr {}",
                            tmp, load_ty, ptr_val
                        )
                        .unwrap();
                        return tmp;
                    }
                    _ => {}
                }

                let ret_ty = if self.fn_llvm_params.contains_key(&func_name) {
                    self.functions
                        .get(&func_name)
                        .map(|(_, ret)| ret.clone())
                        .unwrap_or_else(|| "i32".into())
                } else {
                    match func_name.as_str() {
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
                    }
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
                self.emit_enum_constructor(&full_name, &[]).unwrap_or(full_name)
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
                    // A Q field takes its value in its own format: `0.25` is
                    // `0.25 * 2^frac`, not `fptosi 0.25`.
                    let q_field = self
                        .ast_structs
                        .get(name)
                        .and_then(|fs| fs.iter().find(|(n, _)| n == fname))
                        .and_then(|(_, t)| QFormat::parse(t));
                    if let Some(fmt) = q_field {
                        let raw = self.q_value(fexpr, fmt);
                        let new_val = self.fresh_tmp();
                        writeln!(
                            &mut self.output,
                            "  {} = insertvalue {} {}, {} {}, {}",
                            new_val, ty, current_val, field_ty, raw, field_idx
                        )
                        .unwrap();
                        current_val = new_val;
                        continue;
                    }
                    let mut val = self.emit_expr(fexpr, None, Some(field_ty.clone()));
                    let mut val_ty = self.infer_type(fexpr);
                    if val == "zeroinitializer" {
                        val_ty = field_ty.clone();
                    } else if val_ty.starts_with('[') {
                        // An array evaluates to its address (a local, or an
                        // array-typed field); `insertvalue` wants the value.
                        val = self.emit_load(&val, &val_ty);
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
            // A block expression is the one `Expr` this match did not handle,
            // and the catch-all that stood here turned it into the CONSTANT 0
            // (`add i32 0, 0 ; unhandled expr`). The parser builds none today,
            // so it is refused by name rather than lowered, and the match has
            // no catch-all: a variant added later is a compile error here, not
            // a zero.
            Expr::BlockExpr(_, span) => {
                self.emit_errors.push(format!(
                    "[LLVM host backend] Line {}: a block used as an expression has no lowering \
                     in this backend; it is refused rather than evaluated as 0.",
                    span.line
                ));
                "0".into()
            }
        }
    }

    /// The LLVM type of one element of what a pointer-valued `base` points
    /// at: the stride `base[i]` steps by, and the width it loads and stores.
    ///
    /// This was `"i64"` for every pointer, whatever it pointed at. `Out[i]` on
    /// a `GlobalMemory<I32>` parameter stepped and stored 8 bytes per element,
    /// so a kernel called from host code with a 4-element `I32` array wrote
    /// 16 bytes past it - a wrong answer in a plain build and SIGILL under
    /// `-g`, under "Compilation Successful!" - while `-g` described `Out` as a
    /// pointer to `I32`, so the debugger read the same memory 4 bytes at a
    /// time. `Err` says why the element type is not known, and the caller
    /// refuses rather than guessing a width.
    fn pointer_elem_type(&self, base: &Expr) -> Result<String, String> {
        if let Expr::Ident(name, _) = base {
            if let Some(t) = self.mem_storage_types.get(name) {
                return Ok(t.clone());
            }
        }
        // A reference to an array (`&mut [I32; 4]`), or an array reached
        // through one (`s.buffer` with `s: &mut S`).
        let ast = self.infer_ast_type(base);
        let inner = ast
            .strip_prefix("&mut ")
            .or_else(|| ast.strip_prefix('&'))
            .unwrap_or(&ast);
        if let Some(elem) = inner.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
            match primitive_llvm_type(elem) {
                Some("ptr") => {}
                Some(t) => return Ok(t.to_string()),
                None if self.structs.contains_key(elem) => return Ok(format!("%{}", elem)),
                None => {}
            }
        }
        let what = match base {
            Expr::Ident(name, _) => format!("`{}`", name),
            _ => "this expression".to_string(),
        };
        Err(format!(
            "{} is indexed through a pointer whose element type this backend does not know \
             (its type reads as `{}`)",
            what, ast
        ))
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
            writeln!(
                &mut self.output,
                "  {} = icmp ult i64 {}, {}",
                c, idxs[i], b
            )
            .unwrap();
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
        writeln!(&mut self.output, "  store {} {}, ptr {}", elem_ty, val, dst).unwrap();
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

    fn binop_to_llvm(&self, op: &BinaryOp, ty: &str, unsigned: bool) -> &'static str {
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
                } else if unsigned {
                    "udiv"
                } else {
                    "sdiv"
                }
            }
            BinaryOp::Mod => {
                if is_float {
                    "frem"
                } else if unsigned {
                    "urem"
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
                    "fcmp une"
                } else {
                    "icmp ne"
                }
            }
            BinaryOp::Lt => {
                if is_float {
                    "fcmp olt"
                } else if unsigned {
                    "icmp ult"
                } else {
                    "icmp slt"
                }
            }
            BinaryOp::Gt => {
                if is_float {
                    "fcmp ogt"
                } else if unsigned {
                    "icmp ugt"
                } else {
                    "icmp sgt"
                }
            }
            BinaryOp::Le => {
                if is_float {
                    "fcmp ole"
                } else if unsigned {
                    "icmp ule"
                } else {
                    "icmp sle"
                }
            }
            BinaryOp::Ge => {
                if is_float {
                    "fcmp oge"
                } else if unsigned {
                    "icmp uge"
                } else {
                    "icmp sge"
                }
            }
            BinaryOp::And | BinaryOp::BitAnd => "and",
            BinaryOp::Or | BinaryOp::BitOr => "or",
            BinaryOp::BitXor => "xor",
            BinaryOp::Shl => "shl",
            BinaryOp::Shr => {
                if unsigned {
                    "lshr"
                } else {
                    "ashr"
                }
            }
        }
    }

    fn infer_ast_type(&self, expr: &Expr) -> String {
        if matches!(expr, Expr::BinaryOp { op: BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod, .. }
            | Expr::UnaryOp { op: UnaryOp::Neg, .. })
        {
            if let Some(fmt) = self.q_format(expr) {
                return fmt.name();
            }
        }
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
            Expr::UnaryOp {
                op: UnaryOp::Neg,
                operand,
                ..
            } => self.infer_ast_type(operand),
            Expr::BinaryOp {
                left, op, right, ..
            } => {
                if matches!(
                    op,
                    BinaryOp::Eq
                        | BinaryOp::NotEq
                        | BinaryOp::Lt
                        | BinaryOp::Gt
                        | BinaryOp::Le
                        | BinaryOp::Ge
                        | BinaryOp::And
                        | BinaryOp::Or
                ) {
                    return "bool".into();
                }
                let unsigned = self.binary_is_unsigned(op, left, right);
                match (self.infer_type(expr).as_str(), unsigned) {
                    ("i8", true) => "U8",
                    ("i16", true) => "U16",
                    ("i32", true) => "U32",
                    ("i64", true) => "U64",
                    ("i8", false) => "I8",
                    ("i16", false) => "I16",
                    ("i32", false) => "I32",
                    ("i64", false) => "I64",
                    ("half", _) => "F16",
                    ("float", _) => "F32",
                    ("double", _) => "F64",
                    _ => "Unknown",
                }
                .into()
            }
            Expr::MemberAccess { base, member, .. } => {
                if let Ok(Some((_, layout, index))) = self.enum_payload_field(expr) {
                    return layout.fields[index].1.clone();
                }
                // Approximate base ty
                let base_ty = if let Expr::UnaryOp {
                    op: UnaryOp::Deref,
                    operand,
                    ..
                } = &**base
                {
                    self.infer_ast_type(operand)
                } else {
                    self.infer_ast_type(base)
                };
                let struct_name = base_ty
                    .strip_prefix("&mut ")
                    .or_else(|| base_ty.strip_prefix('&'))
                    .unwrap_or(&base_ty)
                    .trim_start_matches('%');

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
                if let Some(layout) = self.enum_variant_layouts.get(&func_name) {
                    return layout.enum_name.clone();
                }
                if let Some(ret_ast_ty) = self.fn_ast_returns.get(&func_name) {
                    ret_ast_ty.clone()
                } else if let Some((_, ret_ast_ty)) = self.functions.get(&func_name) {
                    ret_ast_ty.clone()
                } else {
                    "Unknown".into()
                }
            }
            Expr::Index { base, .. } => {
                if let Expr::Ident(name, _) = &**base {
                    if let Some(element) = self.mem_ast_types.get(name) {
                        return element.clone();
                    }
                }
                let base_ty = self.infer_ast_type(base);
                let inner = base_ty
                    .strip_prefix("&mut ")
                    .or_else(|| base_ty.strip_prefix('&'))
                    .unwrap_or(&base_ty);
                inner
                    .strip_prefix('[')
                    .and_then(|s| s.strip_suffix(']'))
                    .unwrap_or("Unknown")
                    .to_string()
            }
            Expr::StringLit(_, _) => "String".into(),
            Expr::IntLit(_, _) => "i64".into(),
            Expr::FloatLit(_, _) => "f64".into(),
            Expr::BoolLit(_, _) => "bool".into(),
            Expr::CharLit(_, _) => "char".into(),
            Expr::StructLit { name, .. } => name.clone(),
            Expr::Path { namespace, .. } if self.enums.contains_key(namespace) => namespace.clone(),
            _ => "Unknown".into(),
        }
    }

    fn infer_type(&self, expr: &Expr) -> String {
        if matches!(expr, Expr::BinaryOp { op: BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod, .. }
            | Expr::UnaryOp { op: UnaryOp::Neg, .. })
        {
            if let Some(fmt) = self.q_format(expr) {
                return fmt.llvm();
            }
        }
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
                // Match the value returned by emit_expr, rather than assuming
                // that every directive denotes a fixed-point float.
                if let Some((repr, integer_domain)) = self.zero_drift.get(name) {
                    return if *integer_domain {
                        repr.llvm_type().into()
                    } else {
                        "double".into()
                    };
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
                if self.fn_llvm_params.contains_key(&func_name) {
                    return self
                        .functions
                        .get(&func_name)
                        .map(|(_, ret)| ret.clone())
                        .unwrap_or_else(|| "i32".into());
                }
                if let Some(layout) = self.enum_variant_layouts.get(&func_name) {
                    return self.enum_value_type(layout);
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
                    if matches!(
                        func_name.as_str(),
                        "block_ptr2d_store" | "block_ptr3d_store"
                    ) {
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
            Expr::BinaryOp {
                op, left, right, ..
            } => match op {
                BinaryOp::Eq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::Gt
                | BinaryOp::Le
                | BinaryOp::Ge
                | BinaryOp::And
                | BinaryOp::Or => "i1".into(),
                _ => Self::common_operand_type(&self.infer_type(left), &self.infer_type(right)),
            },
            Expr::MemberAccess { base, member, .. } => {
                if let Ok(Some((_, layout, index))) = self.enum_payload_field(expr) {
                    return layout.fields[index].0.clone();
                }
                let base_ty = if let Expr::UnaryOp {
                    op: UnaryOp::Deref,
                    operand,
                    ..
                } = &**base
                {
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
                    // The width the element's address steps by: one function
                    // for both, or a load reads at a stride it was not
                    // written at. An unknown element is refused where its
                    // address is emitted.
                    self.pointer_elem_type(base)
                        .unwrap_or_else(|_| self.pointee_llvm_type(expr))
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
                    if self.structs.contains_key(name.as_str()) || self.enums.contains_key(name.as_str()) {
                        return Some(format!("%{}", name));
                    }
                }
                None
            }
            Type::Ident(name, _) => {
                if self.structs.contains_key(name.as_str()) || self.enums.contains_key(name.as_str()) {
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
        if clean == "Vec"
            || clean.starts_with("Vec<")
            || clean == "String"
            || clean.starts_with("String<")
            || clean == "Option"
            || clean.starts_with("Option<")
        {
            return "ptr".into();
        }
        if let Some(ty) = primitive_llvm_type(clean) {
            return ty.into();
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
                    if self.ast_structs.contains_key(cleaned) || self.enums.contains_key(cleaned) {
                        return format!("%{}", cleaned);
                    }
                }
                self.pointee_types.get(name).cloned().unwrap_or_else(|| {
                    if let Some(t) = self.locals_ast_type.get(name) {
                        let cleaned = t.trim_start_matches('&').trim_start_matches("mut ");
                        if self.ast_structs.contains_key(cleaned) || self.enums.contains_key(cleaned) {
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
                let base_ty = if let Expr::UnaryOp {
                    op: UnaryOp::Deref,
                    operand,
                    ..
                } = &**base
                {
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
                if let Some(layout) = self.enum_variant_layouts.get(&func_name) {
                    return self.enum_value_type(layout);
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
