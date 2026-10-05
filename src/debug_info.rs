// ============================================================
//  Y — DWARF debug information for the LLVM backend (`-g`)
//  debug_info.rs
// ============================================================
//
//! Source-level debug information, so a Y program can be debugged with
//! `gdb` (or any DWARF debugger) as Y: breakpoints on `.ysu` lines,
//! stepping by Y statement, and Y variables printed with their Y types.
//!
//! # How it is attached
//!
//! The LLVM emitter writes instruction text in roughly two hundred places, and
//! threading a `!dbg` attachment through every one of them would make debug
//! info a property of each call site - the shape in which one site forgets.
//! Instead the emitter writes two kinds of MARKER line, and only when `-g` is
//! on (with it off, the emitted module is byte-for-byte what it was):
//!
//! * [`LOC_MARKER`] `<line> <col> <scope>`: every instruction from here
//!   until the next marker belongs to that source position, in that lexical
//!   scope. `LlvmEmitter::emit_stmt` writes one on entering a statement and
//!   another on leaving it, so the code a compound statement emits AFTER its
//!   body - a `for` loop's increment and back edge, an `if`'s jump to its
//!   merge block - goes back to the compound statement's own line instead of
//!   inheriting the last body statement's.
//! * [`VAR_MARKER`] `<index>`: the stack slot just allocated holds a Y
//!   variable; becomes an `llvm.dbg.declare`.
//!
//! [`DebugInfo::finish`] then walks the finished module once: it attaches
//! `!dbg` to every instruction of every function the emitter registered,
//! turns variable markers into declarations, and appends the metadata.
//! Functions it was not told about - the packed GEMM modules `cpu_gemm`
//! appends - are left exactly as they are.
//!
//! # What the debugger is told, and why it is true
//!
//! This backend keeps every Y binding in a stack slot of its own (allocated
//! in the entry block; `crate::lexical_scope` renames bindings apart so that
//! a shadowing `let` does not share the slot of the binding it shadows), and
//! `-g` compiles at `-O0`, so a variable's slot holds its current value at
//! every statement boundary. That is what makes describing a slot as the
//! variable exact rather than approximate.
//!
//! * **A variable is visible exactly where the language says it exists**:
//!   from the statement after its `let` to the end of its block, a `for`
//!   loop's variable inside the loop, a parameter in the whole function.
//!   Each Y block is a `DILexicalBlock`, and so is the rest of a block after
//!   a `let` - the way rustc describes shadowing - so the debugger reports a
//!   name as unknown before its declaration and after its scope, and resolves
//!   a shadowed name to the inner binding inside the inner scope.
//! * A variable's described type must have the SIZE of the slot it lives in.
//!   Where the declared Y type and the storage disagree (a `Q16.16` outside
//!   `@ZeroDrift` is stored as a plain `i32`), the storage wins: a debugger
//!   reading a declared width that is not the stored one prints garbage, and
//!   debug info that misreports memory is worse than none.
//!
//! The compile unit's language is `DW_LANG_C99`: there is no DWARF code for
//! Y, and C's expression syntax covers what Y's does for `print` - `x + 1`,
//! `v[2]`, `p.x`, `*r`.

use crate::ast::*;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

/// `;@y.dbg.loc <line> <col> <scope>` - a comment, so a module that kept one
/// would still be valid IR; [`DebugInfo::finish`] removes every one. `<scope>`
/// is 0 for the function's own scope and `k + 1` for lexical scope `k`.
pub const LOC_MARKER: &str = ";@y.dbg.loc ";
/// `;@y.dbg.var <index into DebugInfo::vars>`.
pub const VAR_MARKER: &str = ";@y.dbg.var ";

/// Y's gdb extension - pretty-printers and the stack-trace filter - which
/// [`DebugInfo::finish`] embeds in every `-g` build. See the file's header.
pub const GDB_EXTENSION: &str = include_str!("debug_info_gdb.py");
/// The name gdb lists the embedded extension under
/// (`info auto-load python-scripts`).
pub const GDB_EXTENSION_NAME: &str = "ysu-gdb-extension";

/// A Y type as the debugger is told about it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DbgTy {
    /// A two's-complement integer; `name` is the Y spelling the debugger shows.
    Int { name: String, bits: u64, signed: bool },
    Float { name: String, bits: u64 },
    /// One byte holding 0 or 1, which is how LLVM stores an `i1`.
    Bool,
    /// Y `char`: one byte.
    Char,
    /// `&T`, `&mut T`, `GlobalMemory<T>`: a pointer to `T`, or `void *` for an
    /// opaque handle (`Vec`, `Box`, ...).
    Ptr(Option<Box<DbgTy>>),
    /// The runtime's string handle, a pointer to `YStr` (`c_src/runtime.c`).
    Str,
    /// A struct declared in the program.
    Struct(String),
    /// An enum declared in the program.
    Enum(String),
    Array(Box<DbgTy>, u64),
}

/// A function the emitter is defining, keyed by its LLVM symbol.
#[derive(Debug, Clone)]
struct FnInfo {
    /// What the debugger calls it. `fn main` is emitted as `ysu_main` because
    /// the C runtime owns the process's `main`; it is still `main` here.
    display: String,
    /// Index into `DebugInfo::files`.
    file: usize,
    line: usize,
    ret: Option<DbgTy>,
    /// `None` where the parameter's type cannot be described.
    params: Vec<Option<DbgTy>>,
}

/// Where a variable is visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarScope {
    /// Its binding has not been emitted (yet): the `let` is unreachable, or
    /// emission has not got to it. A variable still unbound at the end is
    /// left out - it has no extent the program can stop in.
    Unbound,
    /// The whole function: a parameter.
    Function,
    /// A lexical scope, an index into `DebugInfo::scopes`.
    Block(usize),
}

/// One stack slot that holds a Y variable.
#[derive(Debug, Clone)]
pub struct VarInfo {
    /// The name the source spells, which is what the debugger shows.
    pub name: String,
    /// The LLVM value naming the slot, without the `%` - the binding's own
    /// name, which differs from `name` for a binding that shadows another.
    pub slot: String,
    pub line: usize,
    pub col: usize,
    /// 1-based position for a parameter.
    pub arg: Option<u32>,
    pub ty: DbgTy,
    pub scope: VarScope,
}

/// A lexical scope inside a function: a `{ }` block, a `for` loop, or the
/// rest of a block after a `let`.
#[derive(Debug, Clone)]
struct ScopeInfo {
    /// The enclosing scope; `None` is the function itself.
    parent: Option<usize>,
    line: usize,
    col: usize,
}

struct StructInfo {
    file: usize,
    line: usize,
    /// Every field in declaration order, which is also storage order:
    /// `(name, described type, LLVM storage type)`. Offsets come from the
    /// storage types, because that is the layout the emitted code reads. A
    /// field whose type cannot be described is `None` and is left out of the
    /// struct's description rather than guessed at; the others keep their
    /// true offsets.
    fields: Vec<(String, Option<DbgTy>, String)>,
}

struct EnumInfo {
    file: usize,
    line: usize,
    variants: Vec<String>,
    has_data: bool,
}

/// Debug information for one module, built during emission and attached by
/// [`DebugInfo::finish`].
pub struct DebugInfo {
    /// `(file name, directory)` of every source file; the one being compiled
    /// is first. An `import`ed file's items keep the line numbers of THEIR
    /// file, so each item is attributed to the file it was parsed from.
    files: Vec<(String, String)>,
    /// Item name (an LLVM symbol for a function) -> index into `files`, for
    /// the items an `import` brought in. Everything else is the main file's.
    item_file: HashMap<String, usize>,
    functions: HashMap<String, FnInfo>,
    pub vars: Vec<VarInfo>,
    structs: BTreeMap<String, StructInfo>,
    enums: BTreeMap<String, EnumInfo>,
    /// LLVM field types per struct, as the emitter laid them out.
    ir_structs: BTreeMap<String, Vec<String>>,
    /// The position and scope the emitter last wrote a marker for, so it
    /// writes one only when either changes.
    current: Option<(usize, usize, Option<usize>)>,
    /// Parameters declared so far in the current function.
    next_arg: u32,
    /// Every lexical scope of every function, in creation order.
    scopes: Vec<ScopeInfo>,
    /// The scope instructions emitted now belong to; `None` is the function.
    scope: Option<usize>,
    /// The current function's variables by binding name (`VarInfo::slot`).
    fn_vars: HashMap<String, usize>,
    /// Method symbols (`Point_sum`) and their Y names (`Point::sum`), for the
    /// stack traces the gdb extension prints.
    methods: BTreeMap<String, String>,
    /// The module is compiled with optimisation (`-g -O1` and up), which the
    /// compile unit and every subprogram say, as clang's do.
    optimized: bool,
    /// What the compiler checked, proved or assumed about the program, for
    /// `ydb verify`: `Y_PROGRAM["guarantees"]` in the embedded extension.
    /// `None` when the caller supplied none.
    guarantees: Option<crate::guarantees::Guarantees>,
}

impl DebugInfo {
    /// `source` is the `.ysu` file being compiled. It is made absolute, so the
    /// debugger finds the source whatever directory it is started in.
    pub fn new(source: &std::path::Path) -> Self {
        DebugInfo {
            files: vec![file_entry(source)],
            item_file: HashMap::new(),
            functions: HashMap::new(),
            vars: Vec::new(),
            structs: BTreeMap::new(),
            enums: BTreeMap::new(),
            ir_structs: BTreeMap::new(),
            current: None,
            next_arg: 0,
            scopes: Vec::new(),
            scope: None,
            fn_vars: HashMap::new(),
            methods: BTreeMap::new(),
            optimized: false,
            guarantees: None,
        }
    }

    /// The front end's facts about the program (`TypeChecker::guarantees`
    /// and `require::facts`), which the program then carries.
    pub fn set_guarantees(&mut self, g: crate::guarantees::Guarantees) {
        self.guarantees = Some(g);
    }

    /// A fact the backend established (a substituted kernel), added to the
    /// front end's. Ignored when the caller supplied no guarantees.
    pub fn add_fact(&mut self, f: crate::guarantees::Fact) {
        if let Some(g) = self.guarantees.as_mut() {
            g.facts.push(f);
        }
    }

    /// The `@bounds` the front end took on trust in `item` between two lines.
    pub fn trusted_bounds(&self, item: &str, first: usize, last: usize) -> Vec<crate::guarantees::Assumption> {
        self.guarantees.as_ref().map(|g| g.trusted_bounds(item, first, last)).unwrap_or_default()
    }

    /// The path of the file `item` was parsed from.
    fn file_path(&self, item: &str) -> String {
        let (name, dir) = &self.files[self.file_of(item)];
        std::path::Path::new(dir).join(name).to_string_lossy().into_owned()
    }

    /// The module will be compiled with optimisation.
    pub fn set_optimized(&mut self, optimized: bool) {
        self.optimized = optimized;
    }

    /// `symbol` is the method `display` (`Type::method`).
    pub fn set_method(&mut self, symbol: &str, display: &str) {
        self.methods.insert(symbol.to_string(), display.to_string());
    }

    /// The text embedded in `.debug_gdb_scripts`: the program's own facts the
    /// extension needs, then the extension itself.
    pub fn gdb_script(&self) -> String {
        let enums: Vec<String> = self
            .enums
            .iter()
            .map(|(n, e)| format!("\"{}\": {}", n, if e.has_data { "True" } else { "False" }))
            .collect();
        let methods: Vec<String> =
            self.methods.iter().map(|(s, d)| format!("\"{}\": \"{}\"", s, d)).collect();
        // The guarantees are JSON, which is ASCII here (`json_string`), so
        // inside a Python string literal only `\\` and `"` need escaping.
        let guarantees = match &self.guarantees {
            Some(g) => {
                let json = g.to_json(&|item| self.file_path(item));
                format!("__import__(\"json\").loads(\"{}\")", json.replace('\\', "\\\\").replace('"', "\\\""))
            }
            None => "None".to_string(),
        };
        format!(
            "Y_PROGRAM = {{\"enums\": {{{}}}, \"methods\": {{{}}}, \"guarantees\": {}}}\n{}",
            enums.join(", "),
            methods.join(", "),
            guarantees,
            GDB_EXTENSION
        )
    }

    /// `item` (see [`item_names`]) was parsed from `path`, an `import`ed file.
    pub fn set_item_file(&mut self, item: &str, path: &std::path::Path) {
        let entry = file_entry(path);
        let index = match self.files.iter().position(|f| *f == entry) {
            Some(i) => i,
            None => {
                self.files.push(entry);
                self.files.len() - 1
            }
        };
        self.item_file.insert(item.to_string(), index);
    }

    fn file_of(&self, item: &str) -> usize {
        self.item_file.get(item).copied().unwrap_or(0)
    }

    /// Record the program's structs and enums. `ir_structs` is the emitter's
    /// own field layout (`LlvmEmitter::structs`), so the members are described
    /// at the offsets the code actually uses.
    pub fn register_types(
        &mut self,
        prog: &Program,
        ir_structs: &BTreeMap<String, Vec<(String, String)>>,
    ) {
        for item in &prog.items {
            if let Item::Enum(e) = item {
                self.enums.insert(
                    e.name.clone(),
                    EnumInfo {
                        file: self.file_of(&e.name),
                        line: e.span.line,
                        variants: e.variants.iter().map(|v| v.name.clone()).collect(),
                        has_data: e.variants.iter().any(|v| v.fields.is_some()),
                    },
                );
            }
        }
        for (name, fields) in ir_structs {
            self.ir_structs
                .insert(name.clone(), fields.iter().map(|(_, t)| t.clone()).collect());
        }
        // Field types are resolved once every struct name is known, so a field
        // naming a struct declared later in the file still resolves.
        for item in &prog.items {
            if let Item::Struct(s) = item {
                let ir = match ir_structs.get(&s.name) {
                    Some(f) if f.len() == s.fields.len() => f,
                    _ => continue,
                };
                let fields = s
                    .fields
                    .iter()
                    .zip(ir)
                    .map(|(f, (_, ir_ty))| {
                        (f.name.clone(), self.slot_type(Some(&f.ty), None, ir_ty), ir_ty.clone())
                    })
                    .collect::<Vec<_>>();
                let file = self.file_of(&s.name);
                self.structs.insert(s.name.clone(), StructInfo { file, line: s.span.line, fields });
            }
        }
    }

    /// Start a function. Locations restart at the function's own line.
    pub fn begin_function(
        &mut self,
        symbol: &str,
        display: &str,
        line: usize,
        col: usize,
        params: &[Param],
        ret: Option<&Type>,
    ) {
        let params = params.iter().map(|p| self.from_ast(&p.ty)).collect();
        let ret = ret.and_then(|t| self.from_ast(t));
        let file = self.file_of(symbol);
        self.functions.insert(
            symbol.to_string(),
            FnInfo { display: display.to_string(), file, line, ret, params },
        );
        self.current = Some((line, col, None));
        self.next_arg = 0;
        self.scope = None;
        self.fn_vars.clear();
    }

    /// The position instructions are currently attributed to.
    pub fn current(&self) -> Option<(usize, usize)> {
        self.current.map(|(l, c, _)| (l, c))
    }

    /// Move to `(line, col)` in the current scope. Returns whether that is a
    /// change, i.e. whether the emitter must write a marker.
    pub fn move_to(&mut self, line: usize, col: usize) -> bool {
        let here = Some((line, col, self.scope));
        if self.current == here {
            return false;
        }
        self.current = here;
        true
    }

    /// The `<scope>` field of a marker for the current scope.
    pub fn scope_code(&self) -> usize {
        self.scope.map_or(0, |s| s + 1)
    }

    /// Open a scope inside the current one, starting at `(line, col)`, and
    /// make it current. Returns the scope to restore with [`Self::leave_scope`].
    pub fn enter_scope(&mut self, line: usize, col: usize) -> Option<usize> {
        let outer = self.scope;
        self.scopes.push(ScopeInfo { parent: outer, line, col });
        self.scope = Some(self.scopes.len() - 1);
        outer
    }

    /// Back to `outer`, what [`Self::enter_scope`] returned. Every scope opened
    /// since - a `let`'s included - ends here.
    pub fn leave_scope(&mut self, outer: Option<usize>) {
        self.scope = outer;
    }

    /// The binding `slot` (see [`VarInfo::slot`]) of the current function is
    /// visible from here to the end of the current scope.
    pub fn bind(&mut self, slot: &str) {
        if let Some(&i) = self.fn_vars.get(slot) {
            self.vars[i].scope = match self.scope {
                None => VarScope::Function,
                Some(s) => VarScope::Block(s),
            };
        }
    }

    /// Record a variable's slot; returns the index its marker names, or `None`
    /// when its type cannot be described (it is then left out, not guessed).
    pub fn declare(
        &mut self,
        name: &str,
        line: usize,
        col: usize,
        is_param: bool,
        ty: Option<DbgTy>,
    ) -> Option<usize> {
        let arg = if is_param {
            self.next_arg += 1;
            Some(self.next_arg)
        } else {
            None
        };
        let ty = ty?;
        self.vars.push(VarInfo {
            name: crate::lexical_scope::source_name(name).to_string(),
            slot: name.to_string(),
            line,
            col,
            arg,
            ty,
            // A parameter exists from the function's entry; a `let` or a loop
            // variable from where `bind` is called for it.
            scope: if is_param { VarScope::Function } else { VarScope::Unbound },
        });
        self.fn_vars.insert(name.to_string(), self.vars.len() - 1);
        Some(self.vars.len() - 1)
    }

    // ── Types ───────────────────────────────────────────────

    /// The type a slot is described with: its declared Y type (or the type
    /// name the emitter inferred) when that has the storage's size, else the
    /// storage type itself, else nothing. See the module doc for why the
    /// storage wins a disagreement.
    pub fn slot_type(&self, declared: Option<&Type>, inferred: Option<&str>, storage: &str) -> Option<DbgTy> {
        let stored = ir_size_align(storage, &self.ir_structs, &self.enums).map(|(s, _)| s * 8);
        let wanted = declared
            .and_then(|t| self.from_ast(t))
            .or_else(|| inferred.and_then(|n| self.from_name(n)));
        if let Some(t) = wanted {
            if stored.is_some() && self.size_bits(&t) == stored {
                return Some(t);
            }
        }
        self.from_ir(storage)
    }

    /// A Y type as written in the source.
    pub fn from_ast(&self, ty: &Type) -> Option<DbgTy> {
        match ty {
            Type::Primitive(n, _) | Type::Ident(n, _) => self.from_name(n),
            Type::Reference { inner, .. } => {
                Some(DbgTy::Ptr(self.from_ast(inner).map(Box::new)))
            }
            Type::Generic { base, args, .. } => match base.as_str() {
                "GlobalMemory" | "SharedMemory" => {
                    let inner = match args.first() {
                        Some(GenericArg::Type(t)) => self.from_ast(t),
                        _ => None,
                    };
                    Some(DbgTy::Ptr(inner.map(Box::new)))
                }
                _ => Some(DbgTy::Ptr(None)),
            },
            Type::Array { element, size, .. } => match &**size {
                Expr::IntLit(n, _) if *n > 0 => {
                    Some(DbgTy::Array(Box::new(self.from_ast(element)?), *n as u64))
                }
                _ => None,
            },
            Type::BlockTile { .. } => Some(DbgTy::Ptr(None)),
        }
    }

    /// A type NAME: a primitive, `String`, a struct or enum of the program,
    /// or a reference to one of those (the emitter's inferred types for an
    /// unannotated `let` are names in `ast_type_to_string`'s spelling).
    pub fn from_name(&self, name: &str) -> Option<DbgTy> {
        if let Some(inner) = name.strip_prefix("&mut ").or_else(|| name.strip_prefix('&')) {
            return Some(DbgTy::Ptr(self.from_name(inner).map(Box::new)));
        }
        let int = |n: &str, bits, signed| Some(DbgTy::Int { name: n.into(), bits, signed });
        let float = |n: &str, bits| Some(DbgTy::Float { name: n.into(), bits });
        match name {
            "I8" | "i8" => int("I8", 8, true),
            "I16" | "i16" => int("I16", 16, true),
            "I32" | "i32" => int("I32", 32, true),
            "I64" | "i64" => int("I64", 64, true),
            "isize" => int("isize", 64, true),
            "U8" | "u8" => int("U8", 8, false),
            "U16" | "u16" => int("U16", 16, false),
            "U32" | "u32" => int("U32", 32, false),
            "U64" | "u64" => int("U64", 64, false),
            "usize" => int("usize", 64, false),
            "F16" | "f16" => float("F16", 16),
            "F32" | "f32" => float("F32", 32),
            "F64" | "f64" => float("F64", 64),
            "bool" => Some(DbgTy::Bool),
            "char" => Some(DbgTy::Char),
            "String" => Some(DbgTy::Str),
            "Vec" | "ptr" => Some(DbgTy::Ptr(None)),
            other if self.structs.contains_key(other) || self.ir_structs.contains_key(other) => {
                Some(DbgTy::Struct(other.to_string()))
            }
            other if self.enums.contains_key(other) => Some(DbgTy::Enum(other.to_string())),
            _ => None,
        }
    }

    /// An LLVM storage type, for a slot whose Y type is unknown or does not
    /// fit it. LLVM integers carry no sign, and this backend treats an
    /// unannotated integer as signed, so they are described as signed.
    pub fn from_ir(&self, ir: &str) -> Option<DbgTy> {
        let int = |n: &str, bits| Some(DbgTy::Int { name: n.into(), bits, signed: true });
        match ir {
            "i1" => Some(DbgTy::Bool),
            "i8" => int("I8", 8),
            "i16" => int("I16", 16),
            "i32" => int("I32", 32),
            "i64" => int("I64", 64),
            "half" => Some(DbgTy::Float { name: "F16".into(), bits: 16 }),
            "float" => Some(DbgTy::Float { name: "F32".into(), bits: 32 }),
            "double" => Some(DbgTy::Float { name: "F64".into(), bits: 64 }),
            "ptr" => Some(DbgTy::Ptr(None)),
            _ => {
                if let Some(name) = ir.strip_prefix('%') {
                    if self.enums.contains_key(name) {
                        return Some(DbgTy::Enum(name.to_string()));
                    }
                    if self.ir_structs.contains_key(name) {
                        return Some(DbgTy::Struct(name.to_string()));
                    }
                    return None;
                }
                let (n, elem) = parse_ir_array(ir)?;
                Some(DbgTy::Array(Box::new(self.from_ir(elem)?), n))
            }
        }
    }

    /// Size in bits, or `None` for a type whose layout is not known.
    pub fn size_bits(&self, t: &DbgTy) -> Option<u64> {
        Some(match t {
            DbgTy::Int { bits, .. } | DbgTy::Float { bits, .. } => *bits,
            DbgTy::Bool | DbgTy::Char => 8,
            DbgTy::Ptr(_) | DbgTy::Str => 64,
            DbgTy::Array(e, n) => self.size_bits(e)? * n,
            DbgTy::Struct(n) => {
                ir_size_align(&format!("%{}", n), &self.ir_structs, &self.enums)?.0 * 8
            }
            DbgTy::Enum(n) => {
                if self.enums.get(n)?.has_data {
                    ir_size_align(&format!("%{}", n), &self.ir_structs, &self.enums)?.0 * 8
                } else {
                    32
                }
            }
        })
    }

    // ── Attaching ───────────────────────────────────────────

    /// Attach the debug information to the finished module text.
    pub fn finish(self, module: &str) -> String {
        let mut md = Md::new(max_metadata_id(module) + 1);
        let files: Vec<usize> = self
            .files
            .iter()
            .map(|(name, dir)| {
                md.push(format!(
                    "!DIFile(filename: \"{}\", directory: \"{}\")",
                    md_escape(name),
                    md_escape(dir)
                ))
            })
            .collect();
        let file = files[0];
        // The producer must begin with `clang `: gdb trusts LLVM's line table
        // to find where a function's prologue ends only for a producer it
        // recognises as LLVM (`producer_is_llvm`). For any other it falls back
        // to scanning instructions, finds no frame setup in a -O0 function, and
        // `step` into a call stops on the `fn` line before the parameters are
        // stored - every argument printed as whatever its slot held before.
        // rustc reports `clang LLVM (rustc version ...)` for the same reason.
        let cu = md.push(format!(
            "distinct !DICompileUnit(language: DW_LANG_C99, file: !{}, \
             producer: \"clang LLVM (Y compiler)\", isOptimized: {}, runtimeVersion: 0, \
             emissionKind: FullDebug)",
            file, self.optimized
        ));
        let dwarf = md.push("!{i32 7, !\"Dwarf Version\", i32 4}".into());
        let version = md.push("!{i32 2, !\"Debug Info Version\", i32 3}".into());

        let mut types = TypeNodes { files: files.clone(), ids: HashMap::new(), unspecified: None };
        let mut locs: HashMap<(usize, usize, usize), usize> = HashMap::new();
        let mut out = String::with_capacity(module.len() + module.len() / 2);
        // Inside a described function: its subprogram, its file, and the position the
        // last marker set. No position until the first statement's marker:
        // the entry block's slot allocations and parameter spills are the
        // prologue, and a location on them is where gdb would put a
        // function breakpoint - before the parameters are stored, so every
        // argument would print as whatever the slot held before.
        let mut func: Option<(usize, usize, Option<(usize, usize, Option<usize>)>)> = None;
        let mut declared_intrinsic = false;
        // Lexical scope -> its `DILexicalBlock`, made on first use.
        let mut blocks: HashMap<usize, usize> = HashMap::new();
        // In an optimised build, the current function's variables, kept as
        // the subprogram's `retainedNodes`: otherwise a variable whose every
        // location the optimiser removed is dropped from the debug
        // information, and the debugger says nothing about it rather than
        // `<optimized out>`. Clang does the same.
        let mut retained: Option<(usize, Vec<usize>)> = None;

        for line in module.lines() {
            if let Some(rest) = line.strip_prefix(LOC_MARKER) {
                let mut it = rest.split_whitespace().filter_map(|w| w.parse::<usize>().ok());
                if let (Some((sp, f, _)), Some(l), Some(c), Some(k)) =
                    (func, it.next(), it.next(), it.next())
                {
                    func = Some((sp, f, Some((l, c, k.checked_sub(1)))));
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix(VAR_MARKER) {
                let var = rest.trim().parse::<usize>().ok().and_then(|i| self.vars.get(i));
                if let (Some((sp, vfile, _)), Some(v)) = (func, var) {
                    // A binding that was never emitted has no extent to stop
                    // in; describing it would put it in no scope at all.
                    let scope = match v.scope {
                        VarScope::Unbound => continue,
                        VarScope::Function => sp,
                        VarScope::Block(k) => self.block_md(&mut md, &mut blocks, k, sp, vfile),
                    };
                    let ty = types.node(&self, &mut md, &v.ty);
                    let arg = v.arg.map(|a| format!("arg: {}, ", a)).unwrap_or_default();
                    let var_id = md.push(format!(
                        "!DILocalVariable(name: \"{}\", {}scope: !{}, file: !{}, line: {}, type: {})",
                        md_escape(&v.name),
                        arg,
                        scope,
                        vfile,
                        v.line,
                        ty
                    ));
                    if let Some((_, vars)) = &mut retained {
                        vars.push(var_id);
                    }
                    let loc = location(&mut md, &mut locs, v.line, v.col, scope);
                    writeln!(
                        out,
                        "  call void @llvm.dbg.declare(metadata ptr %{}, metadata !{}, metadata !DIExpression()), !dbg !{}",
                        v.slot, var_id, loc
                    )
                    .unwrap();
                    declared_intrinsic = true;
                }
                continue;
            }
            if line.starts_with("define ") {
                func = None;
                if let Some(f) = define_symbol(line).and_then(|s| self.functions.get(s)) {
                    let ffile = files[f.file];
                    let keep = if self.optimized { Some(md.reserve()) } else { None };
                    retained = keep.map(|id| (id, Vec::new()));
                    let sp = subprogram(&self, &mut md, &mut types, f, ffile, cu, keep);
                    if let Some(brace) = line.rfind('{') {
                        writeln!(out, "{}!dbg !{} {}", &line[..brace], sp, &line[brace..]).unwrap();
                        func = Some((sp, ffile, None));
                        continue;
                    }
                }
            } else if line == "}" {
                func = None;
                if let Some((id, vars)) = retained.take() {
                    let list: Vec<String> = vars.iter().map(|v| format!("!{}", v)).collect();
                    md.define(id, format!("!{{{}}}", list.join(", ")));
                }
            } else if let Some((sp, ffile, Some((l, c, k)))) = func {
                if is_instruction(line) {
                    let scope = match k {
                        None => sp,
                        Some(k) => self.block_md(&mut md, &mut blocks, k, sp, ffile),
                    };
                    let loc = location(&mut md, &mut locs, l, c, scope);
                    out.push_str(&attach_dbg(line, loc));
                    out.push('\n');
                    continue;
                }
            }
            out.push_str(line);
            out.push('\n');
        }

        // The gdb extension, as an inline script gdb auto-loads with the
        // program (section type 4: Python text, a name line, then the text).
        // `@llvm.used` keeps an optimising build from dropping it.
        let script = format!("\u{4}{}\n{}", GDB_EXTENSION_NAME, self.gdb_script());
        writeln!(
            out,
            "\n@__y_debug_gdb_scripts = linkonce_odr unnamed_addr constant [{} x i8] c\"{}\\00\", \
             section \".debug_gdb_scripts\", align 1",
            script.len() + 1,
            md_escape(&script)
        )
        .unwrap();
        if !module.contains("@llvm.used ") {
            out.push_str(
                "@llvm.used = appending global [1 x ptr] [ptr @__y_debug_gdb_scripts], section \"llvm.metadata\"\n",
            );
        }

        out.push_str("\n; --- Debug information (Y -g) ---\n");
        if declared_intrinsic {
            out.push_str("declare void @llvm.dbg.declare(metadata, metadata, metadata)\n");
        }
        writeln!(out, "!llvm.dbg.cu = !{{!{}}}", cu).unwrap();
        writeln!(out, "!llvm.module.flags = !{{!{}, !{}}}", dwarf, version).unwrap();
        for (id, def) in md.defs {
            writeln!(out, "!{} = {}", id, def).unwrap();
        }
        out
    }

    /// The `DILexicalBlock` of scope `k`, whose function's subprogram is `sp`
    /// and file `file`; its enclosing scopes are made first.
    fn block_md(
        &self,
        md: &mut Md,
        blocks: &mut HashMap<usize, usize>,
        k: usize,
        sp: usize,
        file: usize,
    ) -> usize {
        if let Some(&id) = blocks.get(&k) {
            return id;
        }
        let s = &self.scopes[k];
        let parent = match s.parent {
            None => sp,
            Some(p) => self.block_md(md, blocks, p, sp, file),
        };
        let id = md.push(format!(
            "distinct !DILexicalBlock(scope: !{}, file: !{}, line: {}, column: {})",
            parent, file, s.line, s.col
        ));
        blocks.insert(k, id);
        id
    }
}

// ── Metadata numbering ──────────────────────────────────────

/// Numbered metadata, starting above every number the module already uses.
struct Md {
    next: usize,
    defs: Vec<(usize, String)>,
}

impl Md {
    fn new(first: usize) -> Self {
        Md { next: first, defs: Vec::new() }
    }
    fn reserve(&mut self) -> usize {
        self.next += 1;
        self.next - 1
    }
    fn define(&mut self, id: usize, def: String) {
        self.defs.push((id, def));
    }
    fn push(&mut self, def: String) -> usize {
        let id = self.reserve();
        self.define(id, def);
        id
    }
}

fn location(
    md: &mut Md,
    locs: &mut HashMap<(usize, usize, usize), usize>,
    line: usize,
    col: usize,
    scope: usize,
) -> usize {
    *locs.entry((line, col, scope)).or_insert_with(|| {
        md.push(format!("!DILocation(line: {}, column: {}, scope: !{})", line, col, scope))
    })
}

fn subprogram(
    info: &DebugInfo,
    md: &mut Md,
    types: &mut TypeNodes,
    f: &FnInfo,
    file: usize,
    cu: usize,
    retained: Option<usize>,
) -> usize {
    let mut elems = vec![match &f.ret {
        Some(t) => types.node(info, md, t),
        None => "null".to_string(),
    }];
    for p in &f.params {
        elems.push(match p {
            Some(t) => types.node(info, md, t),
            None => types.unspecified(md),
        });
    }
    let sig = md.push(format!("!DISubroutineType(types: !{{{}}})", elems.join(", ")));
    md.push(format!(
        "distinct !DISubprogram(name: \"{}\", scope: !{}, file: !{}, line: {}, type: !{}, \
         scopeLine: {}, flags: DIFlagPrototyped, spFlags: {}, unit: !{}{})",
        md_escape(&f.display),
        file,
        file,
        f.line,
        sig,
        f.line,
        if info.optimized { "DISPFlagDefinition | DISPFlagOptimized" } else { "DISPFlagDefinition" },
        cu,
        retained.map(|r| format!(", retainedNodes: !{}", r)).unwrap_or_default()
    ))
}

/// Type nodes, one per distinct [`DbgTy`].
struct TypeNodes {
    /// Metadata ids of `DebugInfo::files`.
    files: Vec<usize>,
    ids: HashMap<DbgTy, String>,
    unspecified: Option<usize>,
}

impl TypeNodes {
    fn unspecified(&mut self, md: &mut Md) -> String {
        let id = *self.unspecified.get_or_insert_with(|| {
            md.push("!DIBasicType(tag: DW_TAG_unspecified_type, name: \"unknown\")".into())
        });
        format!("!{}", id)
    }

    /// `!N` for `t`, or `null` for `void`.
    fn node(&mut self, info: &DebugInfo, md: &mut Md, t: &DbgTy) -> String {
        if let Some(id) = self.ids.get(t) {
            return id.clone();
        }
        let r = match t {
            DbgTy::Int { name, bits, signed } => format!(
                "!{}",
                md.push(format!(
                    "!DIBasicType(name: \"{}\", size: {}, encoding: {})",
                    md_escape(name),
                    bits,
                    if *signed { "DW_ATE_signed" } else { "DW_ATE_unsigned" }
                ))
            ),
            DbgTy::Float { name, bits } => format!(
                "!{}",
                md.push(format!(
                    "!DIBasicType(name: \"{}\", size: {}, encoding: DW_ATE_float)",
                    md_escape(name),
                    bits
                ))
            ),
            DbgTy::Bool => format!(
                "!{}",
                md.push("!DIBasicType(name: \"bool\", size: 8, encoding: DW_ATE_boolean)".into())
            ),
            DbgTy::Char => format!(
                "!{}",
                md.push("!DIBasicType(name: \"char\", size: 8, encoding: DW_ATE_signed_char)".into())
            ),
            DbgTy::Ptr(None) => format!(
                "!{}",
                md.push("!DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)".into())
            ),
            DbgTy::Ptr(Some(inner)) => {
                // Reserve first: a struct can hold a pointer to itself.
                let id = md.reserve();
                self.ids.insert(t.clone(), format!("!{}", id));
                let base = self.node(info, md, inner);
                md.define(
                    id,
                    format!("!DIDerivedType(tag: DW_TAG_pointer_type, baseType: {}, size: 64)", base),
                );
                return format!("!{}", id);
            }
            DbgTy::Str => {
                // `typedef struct { char* data; int32_t len; int32_t cap; } YStr;`
                // in `c_src/runtime.c`, and a Y `String` is a `YStr*`.
                let chr = self.node(info, md, &DbgTy::Char);
                let i32t = self.node(
                    info,
                    md,
                    &DbgTy::Int { name: "I32".into(), bits: 32, signed: true },
                );
                let data_ty = md.push(format!(
                    "!DIDerivedType(tag: DW_TAG_pointer_type, baseType: {}, size: 64)",
                    chr
                ));
                let ystr = md.reserve();
                let member = |md: &mut Md, name: &str, ty: String, size: u64, off: u64| {
                    md.push(format!(
                        "!DIDerivedType(tag: DW_TAG_member, name: \"{}\", scope: !{}, baseType: {}, \
                         size: {}, offset: {})",
                        name, ystr, ty, size, off
                    ))
                };
                let m_data = member(md, "data", format!("!{}", data_ty), 64, 0);
                let m_len = member(md, "len", i32t.clone(), 32, 64);
                let m_cap = member(md, "cap", i32t, 32, 96);
                md.define(
                    ystr,
                    format!(
                        "!DICompositeType(tag: DW_TAG_structure_type, name: \"YStr\", size: 128, \
                         elements: !{{!{}, !{}, !{}}})",
                        m_data, m_len, m_cap
                    ),
                );
                let ptr = md.push(format!(
                    "!DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{}, size: 64)",
                    ystr
                ));
                format!(
                    "!{}",
                    md.push(format!(
                        "!DIDerivedType(tag: DW_TAG_typedef, name: \"String\", baseType: !{})",
                        ptr
                    ))
                )
            }
            DbgTy::Array(elem, n) => {
                let e = self.node(info, md, elem);
                let range = md.push(format!("!DISubrange(count: {})", n));
                let size = info.size_bits(t).unwrap_or(0);
                format!(
                    "!{}",
                    md.push(format!(
                        "!DICompositeType(tag: DW_TAG_array_type, baseType: {}, size: {}, \
                         elements: !{{!{}}})",
                        e, size, range
                    ))
                )
            }
            DbgTy::Enum(name) => self.enum_node(info, md, t, name),
            DbgTy::Struct(name) => {
                let id = md.reserve();
                self.ids.insert(t.clone(), format!("!{}", id));
                let size = info.size_bits(t).unwrap_or(0);
                let (sfile, line, fields) = match info.structs.get(name) {
                    Some(s) => (self.files[s.file], s.line, s.fields.as_slice()),
                    None => (self.files[0], 0, &[][..]),
                };
                let irs: Vec<String> = fields.iter().map(|(_, _, ir)| ir.clone()).collect();
                let offsets = struct_offsets(&irs, &info.ir_structs, &info.enums);
                let mut members = Vec::new();
                for (i, (fname, fty, _)) in fields.iter().enumerate() {
                    let (Some(fty), Some(off)) = (fty, offsets.get(i).copied()) else {
                        continue;
                    };
                    let Some(sz) = info.size_bits(fty) else { continue };
                    let base = self.node(info, md, fty);
                    members.push(format!(
                        "!{}",
                        md.push(format!(
                            "!DIDerivedType(tag: DW_TAG_member, name: \"{}\", scope: !{}, file: !{}, \
                             line: {}, baseType: {}, size: {}, offset: {})",
                            md_escape(fname),
                            id,
                            sfile,
                            line,
                            base,
                            sz,
                            off * 8
                        ))
                    ));
                }
                md.define(
                    id,
                    format!(
                        "!DICompositeType(tag: DW_TAG_structure_type, name: \"{}\", file: !{}, line: {}, \
                         size: {}, elements: !{{{}}})",
                        md_escape(name),
                        sfile,
                        line,
                        size,
                        members.join(", ")
                    ),
                );
                return format!("!{}", id);
            }
        };
        self.ids.insert(t.clone(), r.clone());
        r
    }

    fn enum_node(&mut self, info: &DebugInfo, md: &mut Md, t: &DbgTy, name: &str) -> String {
        let Some(e) = info.enums.get(name) else {
            return self.unspecified(md);
        };
        let i32t = self.node(info, md, &DbgTy::Int { name: "I32".into(), bits: 32, signed: true });
        let enumerators: Vec<String> = e
            .variants
            .iter()
            .enumerate()
            .map(|(i, v)| {
                format!(
                    "!{}",
                    md.push(format!("!DIEnumerator(name: \"{}\", value: {})", md_escape(v), i))
                )
            })
            .collect();
        let efile = self.files[e.file];
        let tag_name = if e.has_data { format!("{}::tag", name) } else { name.to_string() };
        let tag = md.push(format!(
            "!DICompositeType(tag: DW_TAG_enumeration_type, name: \"{}\", file: !{}, line: {}, \
             baseType: {}, size: 32, elements: !{{{}}})",
            md_escape(&tag_name),
            efile,
            e.line,
            i32t,
            enumerators.join(", ")
        ));
        if !e.has_data {
            let r = format!("!{}", tag);
            self.ids.insert(t.clone(), r.clone());
            return r;
        }
        // A data-carrying enum is `{ i32, [8 x i64] }` in this backend: the
        // tag, then a payload area the variants overlay. The payload is shown
        // as the raw words it is; which variant's fields it holds depends on
        // the tag, which DWARF variant parts could express and gdb's C
        // printing would not use.
        let size = info.size_bits(t).unwrap_or(576);
        let id = md.reserve();
        let words = self.node(
            info,
            md,
            &DbgTy::Array(Box::new(DbgTy::Int { name: "I64".into(), bits: 64, signed: true }), 8),
        );
        let m_tag = md.push(format!(
            "!DIDerivedType(tag: DW_TAG_member, name: \"tag\", scope: !{}, baseType: !{}, size: 32, offset: 0)",
            id, tag
        ));
        let m_payload = md.push(format!(
            "!DIDerivedType(tag: DW_TAG_member, name: \"payload\", scope: !{}, baseType: {}, size: 512, offset: 64)",
            id, words
        ));
        md.define(
            id,
            format!(
                "!DICompositeType(tag: DW_TAG_structure_type, name: \"{}\", file: !{}, line: {}, \
                 size: {}, elements: !{{!{}, !{}}})",
                md_escape(name),
                efile,
                e.line,
                size,
                m_tag,
                m_payload
            ),
        );
        let r = format!("!{}", id);
        self.ids.insert(t.clone(), r.clone());
        r
    }
}

// ── LLVM type layout (x86-64 data layout) ───────────────────

/// `(size, alignment)` in bytes of an LLVM type as this backend writes them,
/// under the data layout `emit_prelude` declares (`i64:64`, natural alignment
/// for every other scalar).
fn ir_size_align(
    ir: &str,
    structs: &BTreeMap<String, Vec<String>>,
    enums: &BTreeMap<String, EnumInfo>,
) -> Option<(u64, u64)> {
    Some(match ir {
        "i1" | "i8" => (1, 1),
        "i16" | "half" => (2, 2),
        "i32" | "float" => (4, 4),
        "i64" | "double" | "ptr" => (8, 8),
        _ => {
            if let Some(name) = ir.strip_prefix('%') {
                if let Some(e) = enums.get(name) {
                    if !e.has_data {
                        return Some((4, 4));
                    }
                    // `%Name = type { i32, [8 x i64] }`
                    return Some((72, 8));
                }
                let fields = structs.get(name)?;
                let mut off = 0u64;
                let mut align = 1u64;
                for f in fields {
                    let (s, a) = ir_size_align(f, structs, enums)?;
                    off = off.div_ceil(a) * a + s;
                    align = align.max(a);
                }
                return Some((off.div_ceil(align) * align, align));
            }
            let (n, elem) = parse_ir_array(ir)?;
            let (s, a) = ir_size_align(elem, structs, enums)?;
            (s * n, a)
        }
    })
}

/// Byte offset of each field, or `u64::MAX` past a field of unknown layout.
fn struct_offsets(
    fields: &[String],
    structs: &BTreeMap<String, Vec<String>>,
    enums: &BTreeMap<String, EnumInfo>,
) -> Vec<u64> {
    let mut off = 0u64;
    let mut out = Vec::new();
    for f in fields {
        match ir_size_align(f, structs, enums) {
            Some((s, a)) => {
                off = off.div_ceil(a) * a;
                out.push(off);
                off += s;
            }
            None => break,
        }
    }
    out
}

/// `[N x T]` -> `(N, "T")`.
fn parse_ir_array(ir: &str) -> Option<(u64, &str)> {
    let inner = ir.strip_prefix('[')?.strip_suffix(']')?;
    let (n, elem) = inner.split_once(" x ")?;
    Some((n.trim().parse().ok()?, elem.trim()))
}

// ── Module text ─────────────────────────────────────────────

/// `(file name, absolute directory)` for a source path, so the debugger finds
/// the source whatever directory it is started in.
fn file_entry(path: &std::path::Path) -> (String, String) {
    let abs = std::fs::canonicalize(path).unwrap_or_else(|_| {
        std::env::current_dir().map(|d| d.join(path)).unwrap_or_else(|_| path.to_path_buf())
    });
    let name = abs
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<input>".into());
    let dir = abs.parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
    (name, dir)
}

/// The names [`DebugInfo::set_item_file`] keys an item by: a function's or
/// kernel's LLVM symbol (an `impl` method is `Type_method`), or a type's name.
pub fn item_names(item: &Item) -> Vec<String> {
    match item {
        Item::Func(f) => vec![f.name.clone()],
        Item::Kernel(k) => vec![k.name.clone()],
        Item::Struct(s) => vec![s.name.clone()],
        Item::Enum(e) => vec![e.name.clone()],
        Item::Impl(imp) => {
            imp.methods.iter().map(|m| format!("{}_{}", imp.target_type, m.name)).collect()
        }
        Item::Import(_) | Item::StaticAssert(_) | Item::Const(_) | Item::Module(_) => Vec::new(),
    }
}

/// The runtime functions that return a string handle (`int32_t` holding a
/// `YStr *` in `c_src/runtime.c`). The emitter infers an unannotated `let`
/// from a call's LLVM return type, which for these is only `ptr`.
pub fn returns_string(callee: &str) -> bool {
    matches!(
        callee,
        "String_new" | "ystr_new" | "ystr_clone" | "String_clone" | "File_read_to_string"
            | "yfile_read_to_string"
    )
}

/// The symbol a `define` line defines: `define i32 @scale(...)` -> `scale`.
fn define_symbol(line: &str) -> Option<&str> {
    let at = line.find('@')?;
    let rest = &line[at + 1..];
    let end = rest.find('(')?;
    Some(&rest[..end])
}

/// An instruction inside a function body: indented, and not a comment. Labels
/// are written at column 0 by this backend, and so are `define` and `}`.
fn is_instruction(line: &str) -> bool {
    line.starts_with("  ") && {
        let t = line.trim_start();
        !t.is_empty() && !t.starts_with(';')
    }
}

/// `line` with `, !dbg !<loc>` appended to the instruction, before any
/// trailing comment. A `;` inside a quoted string (inline asm text) is not a
/// comment; LLVM writes a quote inside a string as `\22`, so a quote always
/// opens or closes one.
fn attach_dbg(line: &str, loc: usize) -> String {
    let mut in_str = false;
    let mut cut = line.len();
    for (i, ch) in line.char_indices() {
        match ch {
            '"' => in_str = !in_str,
            ';' if !in_str => {
                cut = i;
                break;
            }
            _ => {}
        }
    }
    let (code, comment) = line.split_at(cut);
    if comment.is_empty() {
        format!("{}, !dbg !{}", code.trim_end(), loc)
    } else {
        format!("{}, !dbg !{} {}", code.trim_end(), loc, comment)
    }
}

/// The largest `!N` the module defines, so new nodes start above it.
fn max_metadata_id(module: &str) -> usize {
    module
        .lines()
        .filter_map(|l| {
            let rest = l.strip_prefix('!')?;
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() || !rest[digits.len()..].trim_start().starts_with('=') {
                return None;
            }
            digits.parse::<usize>().ok()
        })
        .max()
        .unwrap_or(0)
}

/// A metadata string: printable ASCII as itself, everything else (including
/// `"` and `\`) as LLVM's `\XX` hex escape.
fn md_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if (0x20..0x7f).contains(&b) && b != b'"' && b != b'\\' {
            out.push(b as char);
        } else {
            write!(out, "\\{:02X}", b).unwrap();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trailing_comment_stays_after_the_attachment() {
        assert_eq!(
            attach_dbg("  br i1 %c, label %a, label %b, !uniform_branch !0 ; Maps to X", 7),
            "  br i1 %c, label %a, label %b, !uniform_branch !0, !dbg !7 ; Maps to X"
        );
        assert_eq!(attach_dbg("  %x = add i32 %a, 1", 3), "  %x = add i32 %a, 1, !dbg !3");
    }

    #[test]
    fn a_semicolon_inside_inline_asm_text_is_not_a_comment() {
        assert_eq!(
            attach_dbg("  call void asm sideeffect \"nop; nop\", \"~{memory}\"()", 4),
            "  call void asm sideeffect \"nop; nop\", \"~{memory}\"(), !dbg !4"
        );
    }

    #[test]
    fn struct_layout_matches_the_c_rules_llvm_uses() {
        let mut structs = BTreeMap::new();
        structs.insert(
            "P".to_string(),
            vec!["i32".to_string(), "double".to_string(), "i1".to_string()],
        );
        let enums = BTreeMap::new();
        assert_eq!(ir_size_align("%P", &structs, &enums), Some((24, 8)));
        assert_eq!(struct_offsets(&structs["P"], &structs, &enums), vec![0, 8, 16]);
        assert_eq!(ir_size_align("[3 x i16]", &structs, &enums), Some((6, 2)));
    }

    #[test]
    fn new_metadata_starts_above_the_modules_own() {
        assert_eq!(max_metadata_id("define void @f() {\n}\n!0 = !{i32 1}\n!12 = !{}\n"), 12);
        assert_eq!(max_metadata_id("  br label %x, !uniform_branch !0\n"), 0);
    }

    #[test]
    fn a_path_is_escaped_byte_for_byte() {
        assert_eq!(md_escape("a b\"c\\d"), "a b\\22c\\5Cd");
        assert_eq!(md_escape("ö"), "\\C3\\B6");
    }
}
