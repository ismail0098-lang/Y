// ============================================================
//  Y  —  C11 Backend Emitter
//  c_emitter.rs
//
//  Translates Y AST into portable C11 code. The generated
//  .c file can be compiled by any C11 compiler (gcc, clang,
//  MSVC) to produce a native executable.
//
//  Type mapping:
//    Y         C11
//    ------         ---
//    I32            int32_t
//    I64            int64_t
//    F32            float
//    F64            double
//    String         YStr (custom struct)
//    bool           bool
//    char           char
//    &T             const T*
//    &mut T         T*
//    [T; N]         T name[N]
//    Vec<T>         YVec (void* + element size)
//    Option<T>      tagged struct
// ============================================================

#![allow(dead_code)]

use crate::ast::*;
use std::fmt::Write;

pub struct CEmitter {
    pub output: String,
    indent_level: usize,
    /// Track current impl target for method name mangling
    current_impl_target: Option<String>,
    /// Whether we are inside a @ptx_emit function
    in_ptx_emit: bool,
}

impl CEmitter {
    pub fn new() -> Self {
        Self {
            output: String::new(),
            indent_level: 0,
            current_impl_target: None,
            in_ptx_emit: false,
        }
    }

    fn indent(&mut self) {
        let spaces = "    ".repeat(self.indent_level);
        write!(&mut self.output, "{}", spaces).unwrap();
    }

    fn w(&mut self, s: &str) {
        write!(&mut self.output, "{}", s).unwrap();
    }

    fn wln(&mut self, s: &str) {
        self.indent();
        writeln!(&mut self.output, "{}", s).unwrap();
    }

    // ── Entry Point ─────────────────────────────────────────

    pub fn emit_program(&mut self, prog: &Program) -> String {
        self.emit_prelude();

        // 1. Emit all simple enum definitions first
        for item in &prog.items {
            if let Item::Enum(e) = item {
                let has_data = e.variants.iter().any(|v| v.fields.is_some());
                if !has_data {
                    self.emit_enum(e);
                }
            }
        }

        // 2. Forward-declare all structs
        for item in &prog.items {
            if let Item::Struct(s) = item {
                writeln!(
                    &mut self.output,
                    "typedef struct {name} {name};",
                    name = s.name
                )
                .unwrap();
            }
        }

        // 3. Forward-declare all data-carrying enums (which are compiled as structs)
        for item in &prog.items {
            if let Item::Enum(e) = item {
                let has_data = e.variants.iter().any(|v| v.fields.is_some());
                if has_data {
                    writeln!(
                        &mut self.output,
                        "typedef struct {name} {name};",
                        name = e.name
                    )
                    .unwrap();
                }
            }
        }
        writeln!(&mut self.output).unwrap();

        // 4 & 5. Emit struct and data-carrying enum definitions in their original topological order
        for item in &prog.items {
            match item {
                Item::Struct(s) => self.emit_struct(s),
                Item::Enum(e) => {
                    let has_data = e.variants.iter().any(|v| v.fields.is_some());
                    if has_data {
                        self.emit_enum(e);
                    }
                }
                _ => {}
            }
        }

        // Forward-declare all functions (standalone + impl methods)
        for item in &prog.items {
            match item {
                Item::Func(f) => {
                    let ret = match &f.ret_ty {
                        Some(ty) => self.emit_type(ty),
                        None => "void".into(),
                    };
                    let params: Vec<String> = f
                        .params
                        .iter()
                        .map(|p| format!("{} {}", self.emit_type(&p.ty), p.name))
                        .collect();
                    let params_str = if params.is_empty() {
                        "void".to_string()
                    } else {
                        params.join(", ")
                    };
                    writeln!(&mut self.output, "{} {}({});", ret, f.name, params_str).unwrap();
                }
                Item::Impl(imp) => {
                    for f in &imp.methods {
                        let ret = match &f.ret_ty {
                            Some(ty) => self.emit_type(ty),
                            None => "void".into(),
                        };
                        let params: Vec<String> = f
                            .params
                            .iter()
                            .map(|p| format!("{} {}", self.emit_type(&p.ty), p.name))
                            .collect();
                        let params_str = if params.is_empty() {
                            "void".to_string()
                        } else {
                            params.join(", ")
                        };
                        writeln!(
                            &mut self.output,
                            "{} {}_{}({});",
                            ret, imp.target_type, f.name, params_str
                        )
                        .unwrap();
                    }
                }
                _ => {}
            }
        }
        writeln!(&mut self.output).unwrap();

        // Emit function/impl bodies and remaining items (skip structs/enums already emitted)
        for item in &prog.items {
            match item {
                Item::Struct(_) => {} // already emitted
                Item::Enum(_) => {}   // already emitted
                Item::Func(f) => self.emit_func(f),
                Item::Kernel(k) => self.emit_kernel(k),
                Item::Impl(imp) => self.emit_impl(imp),
                Item::Import(_) => {} // No-op in C
                Item::StaticAssert(sa) => self.emit_static_assert(sa),
                Item::Const(c) => self.emit_const(c),
            }
        }

        self.output.clone()
    }

    // ── C Runtime Prelude ───────────────────────────────────

    fn emit_prelude(&mut self) {
        writeln!(
            &mut self.output,
            r#"/* ============================================== */
/*  Generated by Y Compiler — C11 Backend   */
/* ============================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>

/* ── Y Runtime: String ──────────────────── */
typedef struct {{
    char*  data;
    size_t len;
    size_t cap;
}} YStr;

typedef YStr String;

static uint32_t ychar_to_ascii(uint32_t c) {{
    return c;
}}

static YStr ystr_new(const char* s) {{
    size_t len = s ? strlen(s) : 0;
    size_t cap = len + 1;
    char* data = (char*)malloc(cap);
    if (s) memcpy(data, s, len);
    data[len] = '\0';
    return (YStr){{ data, len, cap }};
}}

typedef union {{
    YStr val;
    YStr* ptr;
}} YStrUnion;

#define YSTR_PTR(x) _Generic((x), \
    YStr*: (x), \
    const YStr*: (x), \
    default: &((YStrUnion){{.val = _Generic((x), YStr: (x), const YStr: (x), default: (YStr){{0}})}}).val \
)

static YStr ystr_clone_impl(const YStr* s) {{
    return ystr_new(s->data);
}}
#define ystr_clone(x) ystr_clone_impl(YSTR_PTR(x))

static void ystr_push_impl(YStr* s, char c) {{
    if (s->len + 1 >= s->cap) {{
        s->cap = s->cap * 2 + 1;
        s->data = (char*)realloc(s->data, s->cap);
    }}
    s->data[s->len++] = c;
    s->data[s->len] = '\0';
}}
#define ystr_push(s, c) ystr_push_impl(&(s), c)

static void ystr_push_str_impl(YStr* s, const YStr* other) {{
    size_t olen = other->len;
    while (s->len + olen >= s->cap) {{
        s->cap = s->cap * 2 + 1;
        s->data = (char*)realloc(s->data, s->cap);
    }}
    memcpy(s->data + s->len, other->data, olen);
    s->len += olen;
    s->data[s->len] = '\0';
}}
#define ystr_push_str(s, other) ystr_push_str_impl(&(s), YSTR_PTR(other))

static bool ystr_eq_impl(const YStr* a, const YStr* b) {{
    if (a->len != b->len) return false;
    return memcmp(a->data, b->data, a->len) == 0;
}}
#define ystr_eq(a, b) ystr_eq_impl(YSTR_PTR(a), YSTR_PTR(b))

#define YSTR_CSTR(x) _Generic(&(x), \
    YStr*: ((const YStr*)(const void*)&(x))->data, \
    const YStr*: ((const YStr*)(const void*)&(x))->data, \
    YStr**: (*(const YStr**)(const void*)&(x))->data, \
    const YStr**: (*(const YStr**)(const void*)&(x))->data, \
    default: (x) \
)

static bool ystr_eq_cstr_impl(const YStr* a, const char* b) {{
    return strcmp(a->data, b) == 0;
}}
#define ystr_eq_cstr(a, b) ystr_eq_cstr_impl(YSTR_PTR(a), YSTR_CSTR(b))

static bool ystr_eq_val_impl(const YStr* a, const YStr* b) {{
    if (a->len != b->len) return false;
    return memcmp(a->data, b->data, a->len) == 0;
}}
#define ystr_eq_val(a, b) ystr_eq_val_impl(YSTR_PTR(a), YSTR_PTR(b))

static char ystr_char_at_impl(const YStr* s, size_t i) {{
    return (i < s->len) ? s->data[i] : '\0';
}}
#define ystr_char_at(s, i) ystr_char_at_impl(YSTR_PTR(s), i)

static void ystr_free_impl(YStr* s) {{
    free(s->data);
    s->data = NULL;
    s->len = 0;
    s->cap = 0;
}}
#define ystr_free(s) ystr_free_impl(&(s))

static size_t ystr_len_impl(const YStr* s) {{
    return s->len;
}}
#define ystr_len(s) ystr_len_impl(YSTR_PTR(s))

/* ── Y Runtime: Dynamic Vector ──────────── */
typedef struct {{
    void*  data;
    size_t len;
    size_t cap;
    size_t elem_size;
}} YVec;

static YVec yvec_new(size_t elem_size) {{
    return (YVec){{ NULL, 0, 0, elem_size }};
}}

typedef union {{
    YVec val;
    YVec* ptr;
}} YVecUnion;

#define YVEC_PTR(x) _Generic((x), \
    YVec*: (x), \
    const YVec*: (x), \
    default: &((YVecUnion){{.val = _Generic((x), YVec: (x), const YVec: (x), default: (YVec){{0}})}}).val \
)

static void yvec_push_impl(YVec* v, const void* elem) {{
    if (v->len >= v->cap) {{
        v->cap = v->cap == 0 ? 8 : v->cap * 2;
        v->data = realloc(v->data, v->cap * v->elem_size);
    }}
    memcpy((char*)v->data + v->len * v->elem_size, elem, v->elem_size);
    v->len++;
}}
#define yvec_push(v, elem) yvec_push_impl(&(v), elem)

static void* yvec_get_impl(const YVec* v, size_t i) {{
    return (char*)v->data + i * v->elem_size;
}}
#define yvec_get(v, i) yvec_get_impl(YVEC_PTR(v), i)

static char yvec_get_char_impl(const YVec* v, size_t i) {{
    return *((char*)yvec_get(v, i));
}}
#define yvec_get_char(v, i) yvec_get_char_impl(YVEC_PTR(v), i)

static size_t yvec_len_impl(const YVec* v) {{
    return v->len;
}}
#define yvec_len(v) yvec_len_impl(YVEC_PTR(v))

static void yvec_free_impl(YVec* v) {{
    free(v->data);
    v->data = NULL;
    v->len = 0;
    v->cap = 0;
}}
#define yvec_free(v) yvec_free_impl(&(v))

/* ── Y Runtime: Print ───────────────────── */
static void yprint_str_impl(const YStr* s) {{
    printf("%s", s->data);
}}
#define yprint_str(s) yprint_str_impl(YSTR_PTR(s))

static void yprint_int(int64_t v) {{
    printf("%lld", (long long)v);
}}

static void yprint_float(double v) {{
    printf("%f", v);
}}

static void yprintln_str_impl(const YStr* s) {{
    printf("%s\n", s->data);
}}
#define yprintln_str(s) yprintln_str_impl(YSTR_PTR(s))

/* ── Y Runtime: File I/O ────────────────── */
static YStr yfile_read_to_string_impl(const YStr* path_str) {{
    const char* path = path_str->data;
    FILE* f = fopen(path, "rb");
    if (!f) {{
        fprintf(stderr, "Error: Cannot open file '%s'\n", path);
        exit(1);
    }}
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = '\0';
    fclose(f);
    YStr result = {{ buf, (size_t)len, (size_t)(len + 1) }};
    return result;
}}
#define yfile_read_to_string(path) yfile_read_to_string_impl(YSTR_PTR(path))

static void yfile_write_impl(const YStr* path_str, const YStr* content) {{
    const char* path = path_str->data;
    FILE* f = fopen(path, "wb");
    if (!f) {{
        fprintf(stderr, "Error: Cannot write file '%s'\n", path);
        exit(1);
    }}
    fwrite(content->data, 1, content->len, f);
    fclose(f);
}}
#define yfile_write(path, content) yfile_write_impl(YSTR_PTR(path), YSTR_PTR(content))

/* ── User Code ───────────────────────────────── */

"#
        )
        .unwrap();
    }

    // ── Types ───────────────────────────────────────────────

    fn emit_type(&self, ty: &Type) -> String {
        match ty {
            Type::Primitive(name, _) => {
                match name.as_str() {
                    "I8" => "int8_t".into(),
                    "I16" => "int16_t".into(),
                    "I32" => "int32_t".into(),
                    "I64" => "int64_t".into(),
                    "U8" => "uint8_t".into(),
                    "U16" => "uint16_t".into(),
                    "U32" => "uint32_t".into(),
                    "U64" => "uint64_t".into(),
                    "F16" => "float".into(), // C has no native f16
                    "F32" => "float".into(),
                    "F64" => "double".into(),
                    "bool" => "bool".into(),
                    "char" => "char".into(),
                    "String" => "YStr".into(),
                    "ptr" => "const char*".into(),
                    other => other.to_string(),
                }
            }
            Type::Ident(name, _) => match name.as_str() {
                "usize" => "size_t".into(),
                "isize" => "ptrdiff_t".into(),
                "Vec" => "YVec".into(),
                "File" => "void*".into(),
                "ptr" => "const char*".into(),
                other => other.to_string(),
            },
            Type::Generic { base, args, .. } => match base.as_str() {
                "Vec" => "YVec".into(),
                "Option" => {
                    if let Some(GenericArg::Type(inner)) = args.first() {
                        format!("YOpt_{}", self.emit_type(inner))
                    } else {
                        "void*".into()
                    }
                }
                "Box" => {
                    if let Some(GenericArg::Type(inner)) = args.first() {
                        format!("{}*", self.emit_type(inner))
                    } else {
                        "void*".into()
                    }
                }
                "GlobalMemory" => "void*".into(),
                "SharedMemory" => "void*".into(),
                _ => format!("{}/*generic*/", base),
            },
            Type::Array { element, .. } => {
                // Arrays are special — handled at declaration site
                // Return element type; caller adds [N]
                self.emit_type(element)
            }
            Type::Reference { mutable, inner, .. } => {
                let inner_str = self.emit_type(inner);
                if *mutable {
                    format!("{}*", inner_str)
                } else {
                    format!("const {}*", inner_str)
                }
            }
        }
    }

    /// Get the array size expression as a string, if the type is an array
    fn array_size_str(&self, ty: &Type) -> Option<String> {
        if let Type::Array { size, .. } = ty {
            Some(self.emit_expr(size))
        } else {
            None
        }
    }

    // ── Struct ──────────────────────────────────────────────

    fn emit_struct(&mut self, s: &StructDecl) {
        writeln!(&mut self.output, "struct {} {{", s.name).unwrap();
        for field in &s.fields {
            let ty_str = self.emit_type(&field.ty);
            let mut prefix = String::new();
            for attr in &field.attrs {
                match &attr.kind {
                    FieldAttrKind::Atomic(_) => {
                        prefix.push_str("_Atomic ");
                    }
                    FieldAttrKind::Align(expr) => {
                        let align_str = self.emit_expr(expr);
                        prefix.push_str(&format!("_Alignas({}) ", align_str));
                    }
                    FieldAttrKind::GpuUncached => {
                        prefix.push_str("volatile ");
                    }
                }
            }
            if let Some(arr_size) = self.array_size_str(&field.ty) {
                writeln!(
                    &mut self.output,
                    "    {}{} {}[{}];",
                    prefix, ty_str, field.name, arr_size
                )
                .unwrap();
            } else {
                writeln!(&mut self.output, "    {}{} {};", prefix, ty_str, field.name).unwrap();
            }
        }
        writeln!(&mut self.output, "}};\n").unwrap();
    }

    // ── Enum ────────────────────────────────────────────────

    fn emit_enum(&mut self, e: &EnumDecl) {
        let has_data = e.variants.iter().any(|v| v.fields.is_some());

        if !has_data {
            // Simple C enum
            writeln!(&mut self.output, "typedef enum {{").unwrap();
            for (i, variant) in e.variants.iter().enumerate() {
                let comma = if i + 1 < e.variants.len() { "," } else { "" };
                writeln!(&mut self.output, "    {}_{}{}", e.name, variant.name, comma).unwrap();
            }
            writeln!(&mut self.output, "}} {};\n", e.name).unwrap();
        } else {
            // Tagged union for data-carrying enums
            // Tag enum
            writeln!(&mut self.output, "typedef enum {{").unwrap();
            for (i, variant) in e.variants.iter().enumerate() {
                let comma = if i + 1 < e.variants.len() { "," } else { "" };
                writeln!(
                    &mut self.output,
                    "    {}_TAG_{}{}",
                    e.name, variant.name, comma
                )
                .unwrap();
            }
            writeln!(&mut self.output, "}} {}_Tag;\n", e.name).unwrap();

            // Data structs for variants with fields
            for variant in &e.variants {
                if let Some(fields) = &variant.fields {
                    writeln!(&mut self.output, "typedef struct {{").unwrap();
                    for (i, field_ty) in fields.iter().enumerate() {
                        let ty_str = self.emit_type(field_ty);
                        writeln!(&mut self.output, "    {} _{};", ty_str, i).unwrap();
                    }
                    writeln!(&mut self.output, "}} {}_{}_Data;\n", e.name, variant.name).unwrap();
                }
            }

            // Main tagged union struct
            writeln!(&mut self.output, "struct {} {{", e.name).unwrap();
            writeln!(&mut self.output, "    {}_Tag tag;", e.name).unwrap();
            writeln!(&mut self.output, "    union {{").unwrap();
            for variant in &e.variants {
                if variant.fields.is_some() {
                    writeln!(
                        &mut self.output,
                        "        {}_{}_Data {};",
                        e.name, variant.name, variant.name
                    )
                    .unwrap();
                }
            }
            writeln!(&mut self.output, "    }} data;").unwrap();
            writeln!(&mut self.output, "}};\n").unwrap();

            // Constructors for data-carrying enums
            for variant in &e.variants {
                if let Some(fields) = &variant.fields {
                    let mut params = Vec::new();
                    let mut assigns = Vec::new();
                    for (i, field_ty) in fields.iter().enumerate() {
                        let ty_str = self.emit_type(field_ty);
                        params.push(format!("{} _{}", ty_str, i));
                        assigns.push(format!("    res.data.{}._{} = _{};", variant.name, i, i));
                    }
                    writeln!(
                        &mut self.output,
                        "static inline {} {}_{}({}) {{",
                        e.name,
                        e.name,
                        variant.name,
                        params.join(", ")
                    )
                    .unwrap();
                    writeln!(&mut self.output, "    {} res;", e.name).unwrap();
                    writeln!(
                        &mut self.output,
                        "    res.tag = {}_TAG_{};",
                        e.name, variant.name
                    )
                    .unwrap();
                    for a in assigns {
                        writeln!(&mut self.output, "{}", a).unwrap();
                    }
                    writeln!(&mut self.output, "    return res;").unwrap();
                    writeln!(&mut self.output, "}}\n").unwrap();
                } else {
                    writeln!(
                        &mut self.output,
                        "#define {}_{} (({}){{ .tag = {}_TAG_{} }})",
                        e.name, variant.name, e.name, e.name, variant.name
                    )
                    .unwrap();
                }
            }
        }
    }

    // ── Functions ───────────────────────────────────────────

    fn emit_func(&mut self, f: &FuncDecl) {
        let prev_ptx = self.in_ptx_emit;
        self.in_ptx_emit = f.is_ptx_emit;

        let ret_type = match &f.ret_ty {
            Some(ty) => self.emit_type(ty),
            None => "void".into(),
        };

        // Build function name (mangled if inside impl)
        let func_name = if let Some(ref target) = self.current_impl_target {
            format!("{}_{}", target, f.name)
        } else {
            f.name.clone()
        };

        // Build parameter list
        let params_str = self.emit_params(&f.params);

        if f.is_ptx_emit {
            writeln!(&mut self.output, "/* @ptx_emit */").unwrap();
        }
        if f.is_ghost {
            writeln!(&mut self.output, "/* @ghost */").unwrap();
        }
        if f.is_hdl_emit {
            writeln!(&mut self.output, "/* @hdl_emit */").unwrap();
        }
        self.indent();
        writeln!(
            &mut self.output,
            "{} {}({}) {{",
            ret_type, func_name, params_str
        )
        .unwrap();
        self.indent_level += 1;
        self.emit_block_body(&f.body);
        self.indent_level -= 1;
        self.wln("}");
        writeln!(&mut self.output).unwrap();

        self.in_ptx_emit = prev_ptx;
    }

    fn emit_params(&self, params: &[Param]) -> String {
        if params.is_empty() {
            return "void".into();
        }
        params
            .iter()
            .map(|p| {
                let ty_str = self.emit_type(&p.ty);
                if let Some(arr_size) = self.array_size_str(&p.ty) {
                    format!("{} {}[{}]", ty_str, p.name, arr_size)
                } else {
                    format!("{} {}", ty_str, p.name)
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn emit_kernel(&mut self, k: &KernelDecl) {
        // Emit kernel as a regular C function (hardware targeting is a future concern)
        let params_str = self.emit_params(&k.params);
        writeln!(&mut self.output, "/* @kernel */").unwrap();
        writeln!(&mut self.output, "void {}({}) {{", k.name, params_str).unwrap();
        self.indent_level += 1;
        self.emit_block_body(&k.body);
        self.indent_level -= 1;
        self.wln("}");
        writeln!(&mut self.output).unwrap();
    }

    fn emit_impl(&mut self, imp: &ImplBlock) {
        writeln!(&mut self.output, "/* impl {} */", imp.target_type).unwrap();
        self.current_impl_target = Some(imp.target_type.clone());
        for method in &imp.methods {
            self.emit_func(method);
        }
        self.current_impl_target = None;
    }

    fn emit_static_assert(&mut self, sa: &StaticAssertDecl) {
        let cond = self.emit_expr(&sa.condition);
        writeln!(
            &mut self.output,
            "_Static_assert({}, \"{}\");\n",
            cond, sa.message
        )
        .unwrap();
    }

    fn emit_const(&mut self, c: &ConstDecl) {
        let ty_str = self.emit_type(&c.ty);
        let val_str = self.emit_expr(&c.value);
        writeln!(
            &mut self.output,
            "static const {} {} = {};\n",
            ty_str, c.name, val_str
        )
        .unwrap();
    }

    // ── Blocks & Statements ─────────────────────────────────

    fn emit_block_body(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.emit_stmt(stmt);
        }
    }

    fn emit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let { name, ty, init, .. } => {
                let ty_str = match ty {
                    Some(t) => self.emit_type(t),
                    None => {
                        // Infer from init or default to auto
                        if let Some(init_expr) = init {
                            self.infer_c_type(init_expr)
                        } else {
                            "int32_t".into()
                        }
                    }
                };

                if let Some(arr_size) = ty.as_ref().and_then(|t| self.array_size_str(t)) {
                    self.indent();
                    if let Some(init_expr) = init {
                        let val = self.emit_expr(init_expr);
                        writeln!(
                            &mut self.output,
                            "{} {}[{}] = {};",
                            ty_str, name, arr_size, val
                        )
                        .unwrap();
                    } else {
                        writeln!(&mut self.output, "{} {}[{}];", ty_str, name, arr_size).unwrap();
                    }
                } else {
                    self.indent();
                    if let Some(init_expr) = init {
                        let val = self.emit_expr(init_expr);
                        writeln!(&mut self.output, "{} {} = {};", ty_str, name, val).unwrap();
                    } else {
                        writeln!(&mut self.output, "{} {};", ty_str, name).unwrap();
                    }
                }
            }
            Stmt::TypeAlias { name, ty, .. } => {
                let ty_str = self.emit_type(ty);
                self.indent();
                writeln!(&mut self.output, "typedef {} {};", ty_str, name).unwrap();
            }
            Stmt::Assign { target, value, .. } => {
                let lhs = self.emit_expr(target);
                let rhs = self.emit_expr(value);
                self.indent();
                writeln!(&mut self.output, "{} = {};", lhs, rhs).unwrap();
            }
            Stmt::CompoundAssign {
                target, op, value, ..
            } => {
                let lhs = self.emit_expr(target);
                let rhs = self.emit_expr(value);
                let op_str = self.binop_to_c(op);
                self.indent();
                writeln!(&mut self.output, "{} {}= {};", lhs, op_str, rhs).unwrap();
            }
            Stmt::Return(expr, _) => {
                self.indent();
                if let Some(e) = expr {
                    let val = self.emit_expr(e);
                    writeln!(&mut self.output, "return {};", val).unwrap();
                } else {
                    writeln!(&mut self.output, "return;").unwrap();
                }
            }
            Stmt::Expr(e) => {
                let val = self.emit_expr(e);
                self.indent();
                writeln!(&mut self.output, "{};", val).unwrap();
            }
            Stmt::For {
                loop_var,
                start,
                end,
                step,
                body,
                ..
            } => {
                let s = self.emit_expr(start);
                let e = self.emit_expr(end);
                let step_str = match step {
                    Some(st) => {
                        let sv = self.emit_expr(st);
                        format!("{} += {}", loop_var, sv)
                    }
                    None => format!("{}++", loop_var),
                };
                self.indent();
                writeln!(
                    &mut self.output,
                    "for (int32_t {} = {}; {} < {}; {}) {{",
                    loop_var, s, loop_var, e, step_str
                )
                .unwrap();
                self.indent_level += 1;
                self.emit_block_body(body);
                self.indent_level -= 1;
                self.wln("}");
            }
            Stmt::If {
                condition,
                then_block,
                else_block,
                ..
            } => {
                let cond = self.emit_expr(condition);
                self.indent();
                writeln!(&mut self.output, "if ({}) {{", cond).unwrap();
                self.indent_level += 1;
                self.emit_block_body(then_block);
                self.indent_level -= 1;
                if let Some(eb) = else_block {
                    // Check if the else block contains a single If stmt (else if chain)
                    if eb.stmts.len() == 1 {
                        if let Stmt::If { .. } = &eb.stmts[0] {
                            self.indent();
                            write!(&mut self.output, "}} else ").unwrap();
                            // Emit the inner if without indent (it will add its own)
                            self.emit_stmt(&eb.stmts[0]);
                            return;
                        }
                    }
                    self.indent();
                    writeln!(&mut self.output, "}} else {{").unwrap();
                    self.indent_level += 1;
                    self.emit_block_body(eb);
                    self.indent_level -= 1;
                    self.wln("}");
                } else {
                    self.wln("}");
                }
            }
            Stmt::While {
                condition, body, ..
            } => {
                let cond = self.emit_expr(condition);
                self.indent();
                writeln!(&mut self.output, "while ({}) {{", cond).unwrap();
                self.indent_level += 1;
                self.emit_block_body(body);
                self.indent_level -= 1;
                self.wln("}");
            }
            Stmt::Match {
                scrutinee, arms, ..
            } => {
                let scrut = self.emit_expr(scrutinee);
                self.indent();
                writeln!(&mut self.output, "/* match {} */", scrut).unwrap();
                // Emit as if-else chain (C switch can't handle arbitrary patterns)
                for (i, arm) in arms.iter().enumerate() {
                    let prefix = if i == 0 { "if" } else { "} else if" };
                    match &arm.pattern {
                        MatchPattern::Wildcard(_) => {
                            if i == 0 {
                                self.indent();
                                writeln!(&mut self.output, "/* default */ {{").unwrap();
                            } else {
                                self.indent();
                                writeln!(&mut self.output, "}} else {{").unwrap();
                            }
                        }
                        MatchPattern::Ident(name, _) => {
                            self.indent();
                            writeln!(&mut self.output, "{} ({} == {}) {{", prefix, scrut, name)
                                .unwrap();
                        }
                        MatchPattern::Literal(lit) => {
                            let lit_str = self.emit_expr(lit);
                            self.indent();
                            writeln!(&mut self.output, "{} ({} == {}) {{", prefix, scrut, lit_str)
                                .unwrap();
                        }
                        MatchPattern::EnumVariant { path, variant, .. } => {
                            let tag = if path.is_empty() {
                                format!("{}", variant)
                            } else {
                                format!("{}_TAG_{}", path, variant)
                            };
                            self.indent();
                            writeln!(&mut self.output, "{} ({}.tag == {}) {{", prefix, scrut, tag)
                                .unwrap();
                        }
                    }
                    self.indent_level += 1;
                    let body = self.emit_expr(&arm.body);
                    self.indent();
                    writeln!(&mut self.output, "{};", body).unwrap();
                    self.indent_level -= 1;
                }
                if !arms.is_empty() {
                    self.wln("}");
                }
            }
            Stmt::Chisel(block, _) => {
                if self.in_ptx_emit {
                    self.wln("/* chisel block — PTX inline assembly */");
                    self.wln("#if defined(__NVCC__) || defined(__CUDACC__)");
                    self.wln("{");
                    self.indent_level += 1;
                    for stmt in &block.stmts {
                        if let Stmt::Expr(Expr::StringLit(s, _)) = stmt {
                            self.indent();
                            writeln!(&mut self.output, "asm volatile (\"{}\");", s).unwrap();
                        } else {
                            self.emit_stmt(stmt);
                        }
                    }
                    self.indent_level -= 1;
                    self.wln("}");
                    self.wln("#else");
                    self.wln("/* PTX asm elided on non-CUDA host target */");
                    for stmt in &block.stmts {
                        if let Stmt::Expr(Expr::StringLit(s, _)) = stmt {
                            self.indent();
                            writeln!(&mut self.output, "/* PTX: {} */", s).unwrap();
                        }
                    }
                    self.wln("#endif");

                } else {
                    self.wln("/* chisel block — privileged hardware ops */");
                    self.wln("{");
                    self.indent_level += 1;
                    self.emit_block_body(block);
                    self.indent_level -= 1;
                    self.wln("}");
                }
            }
            Stmt::SafeBlock(block, _) => {
                self.wln("/* @safe verified block */");
                self.wln("{");
                self.indent_level += 1;
                self.emit_block_body(block);
                self.indent_level -= 1;
                self.wln("}");
            }
            Stmt::GhostBlock(block, _) => {
                self.wln("/* @ghost speculative block */");
                self.wln("{");
                self.indent_level += 1;
                self.emit_block_body(block);
                self.indent_level -= 1;
                self.wln("}");
            }
            Stmt::ClockDomainBlock { body, .. } => {
                self.wln("/* @clock_domain block */");
                self.wln("{");
                self.indent_level += 1;
                self.emit_block_body(body);
                self.indent_level -= 1;
                self.wln("}");
            }
            Stmt::CompileTimeAssert { .. } => {
                self.wln("/* compile_time::assert! verified and stripped */");
            }
        }
    }

    // ── Expressions ─────────────────────────────────────────

    fn emit_expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::IntLit(v, _) => format!("{}", v),
            Expr::FloatLit(v, _) => {
                if *v == (*v as i64) as f64 {
                    format!("{:.1}", v) // Ensure 0.0 not 0
                } else {
                    format!("{}", v)
                }
            }
            Expr::CharLit(c, _) => {
                if *c == '\n' {
                    "'\\n'".into()
                } else if *c == '\r' {
                    "'\\r'".into()
                } else if *c == '\t' {
                    "'\\t'".into()
                } else if *c == '\0' {
                    "'\\0'".into()
                } else if *c == '\'' {
                    "'\\''".into()
                } else if *c == '\\' {
                    "'\\\\'".into()
                } else {
                    format!("'{}'", c)
                }
            }
            Expr::StringLit(s, _) => format!(
                "ystr_new(\"{}\")",
                s.replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
                    .replace('\t', "\\t")
            ),
            Expr::BoolLit(b, _) => {
                if *b {
                    "true".into()
                } else {
                    "false".into()
                }
            }
            Expr::SelfLit(_) => "self".into(),
            Expr::Ident(name, _) => name.clone(),
            Expr::Path {
                namespace, member, ..
            } => {
                format!("{}_{}", namespace, member)
            }
            Expr::BinaryOp {
                left, op, right, ..
            } => {
                let l = self.emit_expr(left);
                let r = self.emit_expr(right);
                let op_str = self.binop_to_c(op);
                format!("({} {} {})", l, op_str, r)
            }
            Expr::UnaryOp { op, operand, .. } => {
                let o = self.emit_expr(operand);
                match op {
                    UnaryOp::Neg => format!("(-{})", o),
                    UnaryOp::Not => format!("(!{})", o),
                    UnaryOp::Ref => {
                        if o.contains('(') && !o.starts_with('(') {
                            // Taking address of a function call result (rvalue).
                            // Use compound literal to ensure block lifetime:
                            format!("&((YStrUnion){{.val = {}}}).val", o)
                        } else {
                            format!("(&{})", o)
                        }
                    }
                    UnaryOp::Deref => format!("(*{})", o),
                }
            }
            Expr::Call { func, args, .. } => {
                let func_str = self.emit_expr(func);
                let args_str: Vec<String> = args.iter().map(|a| self.emit_expr(a)).collect();
                // Map Y built-in functions to C equivalents
                match func_str.as_str() {
                    "println" => {
                        if args_str.is_empty() {
                            "printf(\"\\n\")".into()
                        } else {
                            format!("yprintln_str({})", args_str[0])
                        }
                    }
                    "print" => {
                        if args_str.is_empty() {
                            "((void)0)".into()
                        } else {
                            format!("yprint_str({})", args_str[0])
                        }
                    }
                    "print_int" => format!("printf(\"%d\\n\", (int){})", args_str.join(", ")),
                    "print_float" => format!("printf(\"%f\\n\", {})", args_str.join(", ")),
                    "File_read" | "yfile_read" => {
                        format!("yfile_read_to_string({})", args_str.join(", "))
                    }
                    "File_write" | "yfile_write" => format!(
                        "yfile_write({}, {})",
                        args_str.get(0).unwrap_or(&"\"\"".into()),
                        args_str.get(1).unwrap_or(&"\"\"".into())
                    ),

                    // Bootstrapping type mappings
                    "String_new" => args_str.get(0).unwrap_or(&"ystr_new(\"\")".into()).clone(),
                    "String_clone" => format!("ystr_clone({})", args_str[0]),
                    "String_len" => format!("ystr_len({})", args_str[0]),
                    "String_eq" => format!("ystr_eq({}, {})", args_str[0], args_str[1]),
                    "String_eq_cstr" => {
                        if args_str[1].contains('"') {
                            let raw = args_str[1]
                                .trim_start_matches("ystr_new(")
                                .trim_end_matches(')');
                            format!("ystr_eq_cstr({}, {})", args_str[0], raw)
                        } else {
                            format!("ystr_eq_cstr({}, {})", args_str[0], args_str[1])
                        }
                    }
                    "String_char_at" => format!("ystr_char_at({}, {})", args_str[0], args_str[1]),
                    "String_push" => {
                        let ptr_arg = if args_str[0].starts_with('&') || args_str[0].starts_with("(&") || args_str[0] == "self" {
                            args_str[0].clone()
                        } else {
                            format!("&{}", args_str[0])
                        };
                        format!("ystr_push_impl({}, {})", ptr_arg, args_str[1])
                    }
                    "String_push_str" => {
                        let ptr_arg = if args_str[0].starts_with('&') || args_str[0].starts_with("(&") || args_str[0] == "self" {
                            args_str[0].clone()
                        } else {
                            format!("&{}", args_str[0])
                        };
                        format!("ystr_push_str_impl({}, YSTR_PTR({}))", ptr_arg, args_str[1])
                    }
                    "String_free" => {
                        let ptr_arg = if args_str[0].starts_with('&') || args_str[0].starts_with("(&") || args_str[0] == "self" {
                            args_str[0].clone()
                        } else {
                            format!("&{}", args_str[0])
                        };
                        format!("ystr_free_impl({})", ptr_arg)
                    }
                    "Vec_new" => format!("yvec_new({})", args_str.get(0).unwrap_or(&"1".into())),
                    "Vec_push" => {
                        let ptr_arg = if args_str[0].starts_with('&') || args_str[0].starts_with("(&") || args_str[0] == "self" {
                            args_str[0].clone()
                        } else {
                            format!("&{}", args_str[0])
                        };
                        let val_arg = &args_str[1];
                        if val_arg.starts_with('&') || val_arg.starts_with("(&") {
                            format!("yvec_push_impl({}, {})", ptr_arg, val_arg)
                        } else {
                            format!(
                                "{{ __typeof__({1}) __push_tmp = {1}; yvec_push_impl({0}, &__push_tmp); }}",
                                ptr_arg, val_arg
                            )
                        }
                    }
                    "Vec_get_char" => format!("yvec_get_char({}, {})", args_str[0], args_str[1]),
                    "Vec_get_Token" => {
                        format!("(*(Token*)yvec_get({}, {}))", args_str[0], args_str[1])
                    }
                    "Vec_get_Expr" => {
                        format!("(*(Expr*)yvec_get({}, {}))", args_str[0], args_str[1])
                    }
                    "Vec_get_Stmt" => {
                        format!("(*(Stmt*)yvec_get({}, {}))", args_str[0], args_str[1])
                    }
                    "Vec_get_FuncDecl" => {
                        format!("(*(FuncDecl*)yvec_get({}, {}))", args_str[0], args_str[1])
                    }
                    "Vec_get_ParamDecl" => {
                        format!("(*(ParamDecl*)yvec_get({}, {}))", args_str[0], args_str[1])
                    }
                    "Vec_get_usize" => {
                        format!("(*(size_t*)yvec_get({}, {}))", args_str[0], args_str[1])
                    }
                    "Vec_len" => format!("yvec_len({})", args_str[0]),

                    _ => {
                        if func_str.starts_with("Vec_get_") {
                            let type_name = &func_str["Vec_get_".len()..];
                            let c_type = match type_name {
                                "usize" => "size_t",
                                "I8" | "i8" => "int8_t",
                                "U8" | "u8" => "uint8_t",
                                "I16" | "i16" => "int16_t",
                                "U16" | "u16" => "uint16_t",
                                "I32" | "i32" => "int32_t",
                                "U32" | "u32" => "uint32_t",
                                "I64" | "i64" => "int64_t",
                                "U64" | "u64" => "uint64_t",
                                _ => type_name,
                            };
                            format!("(*({}*)yvec_get({}, {}))", c_type, args_str[0], args_str[1])
                        } else {
                            format!("{}({})", func_str, args_str.join(", "))
                        }
                    }
                }
            }
            Expr::GenericCall { func, args, .. } => {
                // In C, generics don't exist — just emit the function call
                let func_str = self.emit_expr(func);
                let args_str: Vec<String> = args.iter().map(|a| self.emit_expr(a)).collect();
                format!("{}({})", func_str, args_str.join(", "))
            }
            Expr::Index { base, index, .. } => {
                let b = self.emit_expr(base);
                let i = self.emit_expr(index);
                format!("{}[{}]", b, i)
            }
            Expr::MemberAccess { base, member, .. } => {
                let b = self.emit_expr(base);
                format!("{}.{}", b, member)
            }
            Expr::BlockExpr(block, _) => {
                // GCC extension: statement expression ({ stmts; expr; })
                // For portability, just emit last expression
                if let Some(last) = block.stmts.last() {
                    if let Stmt::Expr(e) = last {
                        return self.emit_expr(e);
                    }
                }
                "((void)0)".into()
            }
            Expr::ZeroInit(_) => "{0}".into(),
            Expr::StructLit { name, fields, .. } => {
                let mut assigns = Vec::new();
                for (fname, fexpr) in fields {
                    let val = self.emit_expr(fexpr);
                    assigns.push(format!(".{} = {}", fname, val));
                }
                format!("({}){{ {} }}", name, assigns.join(", "))
            }
        }
    }

    fn binop_to_c(&self, op: &BinaryOp) -> &'static str {
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

    /// Simple C type inference for untyped let bindings
    fn infer_c_type(&self, expr: &Expr) -> String {
        match expr {
            Expr::IntLit(..) => "int64_t".into(),
            Expr::FloatLit(..) => "double".into(),
            Expr::StringLit(..) => "YStr".into(),
            Expr::BoolLit(..) => "bool".into(),
            Expr::Call { func, .. } => {
                // Try to guess from function name
                let name = self.emit_expr(func);
                if name.contains("ystr") || name.contains("read") {
                    "YStr".into()
                } else {
                    "int32_t".into()
                }
            }
            _ => "int32_t".into(), // Conservative default
        }
    }
}

