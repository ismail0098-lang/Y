// ============================================================
//  Y  —  Abstract Syntax Tree (AST)
//  ast.rs
//
//  Defines the hierarchical structures for Y source
//  code, mapping directly from the hardware-sentient spec.
//  The architecture target is deeply integrated but dynamic
//  to support Ada Lovelace, Hopper, and CPU fallbacks.
// ============================================================

#![allow(dead_code)]

/// Represents a source code location.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub line: usize,
    pub col: usize,
}

/// A parsed Y Program.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<Item>,
}

/// A top-level construct in Y (currently mostly Kernels).
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Kernel(KernelDecl),
    Func(FuncDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    Import(ImportDecl),
    StaticAssert(StaticAssertDecl),
    Impl(ImplBlock),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncDecl {
    pub name: String,
    pub is_safe: bool,
    pub params: Vec<Param>,
    pub ret_ty: Option<Type>,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDecl {
    pub name: String,
    pub generic_params: Vec<GenericParam>,
    pub fields: Vec<Field>,
    pub span: Span,
}

/// A generic parameter in a struct/type declaration, e.g., `T` or `SIZE`.
#[derive(Debug, Clone, PartialEq)]
pub struct GenericParam {
    pub name: String,
    pub span: Span,
}

/// Attribute applied to a struct field (e.g., `@gpu_uncached`, `@atomic`, `@align(...)`).
#[derive(Debug, Clone, PartialEq)]
pub enum FieldAttrKind {
    GpuUncached,
    Atomic,
    Align(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldAttr {
    pub kind: FieldAttrKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub attrs: Vec<FieldAttr>,
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumDecl {
    pub name: String,
    pub generic_params: Vec<GenericParam>,
    pub variants: Vec<EnumVariant>,
    pub span: Span,
}

/// A single variant in an enum, optionally carrying data.
/// e.g., `Eof` or `IntLit(i64)` or `Named { line: u32, col: u32 }`
#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: String,
    pub fields: Option<Vec<Type>>, // None = unit, Some = tuple variant
    pub span: Span,
}

/// `impl TypeName { fn methods... }`
#[derive(Debug, Clone, PartialEq)]
pub struct ImplBlock {
    pub target_type: String,
    pub generic_params: Vec<GenericParam>,
    pub methods: Vec<FuncDecl>,
    pub span: Span,
}

/// `import std.mmu;`
#[derive(Debug, Clone, PartialEq)]
pub struct ImportDecl {
    pub path: Vec<String>,
    pub span: Span,
}

/// `@static_assert(condition, "message");`
#[derive(Debug, Clone, PartialEq)]
pub struct StaticAssertDecl {
    pub condition: Expr,
    pub message: String,
    pub span: Span,
}

/// `@require(condition)` to specify hardware requirements.
#[derive(Debug, Clone, PartialEq)]
pub struct RequireAttr {
    pub condition: Expr,
    pub span: Span,
}

/// Defines a Kernel, e.g. `kernel matmul(A: GlobalMemory<F16>...) { ... }`
#[derive(Debug, Clone, PartialEq)]
pub struct KernelDecl {
    pub requires: Vec<RequireAttr>,
    pub name: String,
    pub params: Vec<Param>,
    pub body: Block,
    pub span: Span,
}

/// A parameter in a kernel definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

/// A block of statements enclosed in `{ ... }`
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

// ── Statements ──────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// `let x: T = expr;`
    Let {
        name: String,
        ty: Option<Type>,
        init: Option<Expr>,
        cache_policy: Option<CachePolicyAttr>,
        zero_drift: Option<ZeroDriftAttr>,
        bounds: Option<BoundsAttr>,
        span: Span,
    },
    /// `type ATile = SmemLayout<...>;`
    TypeAlias { name: String, ty: Type, span: Span },
    /// `for k in 0..K step 16 { ... }`
    For {
        loop_var: String,
        start: Expr,
        end: Expr,
        step: Option<Expr>,
        body: Block,
        invariant: Option<Box<Expr>>,
        is_uniform_branch: bool,
        span: Span,
    },
    /// `acc = expr;`
    Assign {
        target: Expr,
        value: Expr,
        span: Span,
    },
    /// A standalone expression evaluated for side effects (e.g., `pipe.wait(tx);`)
    Expr(Expr),
    /// `return expr;`
    Return(Option<Expr>, Span),
    /// `chisel { ... }` — privileged hardware-access block
    Chisel(Block, Span),
    /// `if condition { ... } else { ... }`
    If {
        condition: Box<Expr>,
        then_block: Block,
        else_block: Option<Block>,
        is_uniform_branch: bool,
        span: Span,
    },
    /// `while condition { ... }`
    While {
        condition: Box<Expr>,
        body: Block,
        invariant: Option<Box<Expr>>,
        is_uniform_branch: bool,
        span: Span,
    },
    /// `match scrutinee { pattern => body, ... }`
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
        span: Span,
    },
    /// `target += value;` (and -=, *=, /=)
    CompoundAssign {
        target: Expr,
        op: BinaryOp,
        value: Expr,
        span: Span,
    },
    /// `@safe { ... }` — strict verification block
    SafeBlock(Block, Span),
}

impl Stmt {
    pub fn span(&self) -> Span {
        match self {
            Stmt::Let { span, .. } => span.clone(),
            Stmt::TypeAlias { span, .. } => span.clone(),
            Stmt::For { span, .. } => span.clone(),
            Stmt::Assign { span, .. } => span.clone(),
            Stmt::Expr(e) => e.span(),
            Stmt::Return(_, s) => s.clone(),
            Stmt::Chisel(_, s) => s.clone(),
            Stmt::If { span, .. } => span.clone(),
            Stmt::While { span, .. } => span.clone(),
            Stmt::Match { span, .. } => span.clone(),
            Stmt::CompoundAssign { span, .. } => span.clone(),
            Stmt::SafeBlock(_, s) => s.clone(),
        }
    }
}

/// A single arm of a match expression: `pattern => body`
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: MatchPattern,
    pub body: Expr,
    pub span: Span,
}

/// Pattern for match arms (simplified for bootstrap)
#[derive(Debug, Clone, PartialEq)]
pub enum MatchPattern {
    /// `Ident` — matches an enum variant or binds a variable
    Ident(String, Span),
    /// `SomeEnum::Variant(binding)` — destructuring
    EnumVariant {
        path: String,
        variant: String,
        bindings: Vec<String>,
        span: Span,
    },
    /// `42` or `"hello"` — literal match
    Literal(Expr),
    /// `_` — wildcard
    Wildcard(Span),
}

// ── Expressions ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Ident(String, Span),
    IntLit(i64, Span),
    FloatLit(f64, Span),
    StringLit(String, Span),
    CharLit(char, Span),
    /// `expr1(expr2, ...)` or `cp_async(...)`
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    /// `SharedMemory::alloc<ATile>()`
    GenericCall {
        func: Box<Expr>,
        generic_args: Vec<Type>,
        args: Vec<Expr>,
        span: Span,
    },
    /// `A[k]`
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    /// `pipe.wait`
    MemberAccess {
        base: Box<Expr>,
        member: String,
        span: Span,
    },
    /// `Namespace::Function`
    Path {
        namespace: String,
        member: String,
        span: Span,
    },
    /// `true` / `false`
    BoolLit(bool, Span),
    /// `a + b`, `x == y`, `a && b`
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
        span: Span,
    },
    /// `-x`, `!flag`
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expr>,
        span: Span,
    },
    /// Block expression `{ stmts; final_expr }`
    BlockExpr(Block, Span),
    /// `self`
    SelfLit(Span),
    /// Struct instantiation: `Token { kind: Eof, line: 1 }`
    StructLit {
        name: String,
        fields: Vec<(String, Box<Expr>)>,
        span: Span,
    },
    /// Zero-initialization: `{}`
    ZeroInit(Span),
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::ZeroInit(s) => s.clone(),
            Expr::Ident(_, s) => s.clone(),
            Expr::IntLit(_, s) => s.clone(),
            Expr::FloatLit(_, s) => s.clone(),
            Expr::StringLit(_, s) => s.clone(),
            Expr::CharLit(_, s) => s.clone(),
            Expr::BoolLit(_, s) => s.clone(),
            Expr::SelfLit(s) => s.clone(),
            Expr::Call { span, .. } => span.clone(),
            Expr::GenericCall { span, .. } => span.clone(),
            Expr::Index { span, .. } => span.clone(),
            Expr::MemberAccess { span, .. } => span.clone(),
            Expr::Path { span, .. } => span.clone(),
            Expr::BinaryOp { span, .. } => span.clone(),
            Expr::UnaryOp { span, .. } => span.clone(),
            Expr::BlockExpr(_, s) => s.clone(),
            Expr::StructLit { span, .. } => span.clone(),
        }
    }
}

// ── Types ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Primitive types (F16, F32, I32, u32)
    Primitive(String, Span),
    /// Unresolved generic base types `GlobalMemory<F16>`, `SmemLayout<...>`
    Generic {
        base: String,
        args: Vec<GenericArg>,
        span: Span,
    },
    /// Single token type identifiers
    Ident(String, Span),
    /// Array type `[T; SIZE]`
    Array {
        element: Box<Type>,
        size: Box<Expr>,
        span: Span,
    },
    /// Reference type: `&T` or `&mut T`
    Reference {
        mutable: bool,
        inner: Box<Type>,
        span: Span,
    },
}

// ── Operators ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,    // +
    Sub,    // -
    Mul,    // *
    Div,    // /
    Mod,    // %
    Eq,     // ==
    NotEq,  // !=
    Lt,     // <
    Gt,     // >
    Le,     // <=
    Ge,     // >=
    And,    // &&
    Or,     // ||
    BitAnd, // &
    BitOr,  // |
    BitXor, // ^
    Shl,    // <<
    Shr,    // >>
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg,   // -
    Not,   // !
    Ref,   // &
    Deref, // *
}

/// Generic arguments can be Types (`F16`), Values (`3`), or Named (`rows=16`)
#[derive(Debug, Clone, PartialEq)]
pub enum GenericArg {
    Type(Type),
    Value(Expr),
    Named { name: String, val: Expr },
}

// ── Attributes ──────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct CachePolicyAttr {
    pub policy: String, // "L2_PERSIST"
    pub reuse_count: Option<i64>,
    pub span: Span,
}

/// `@ZeroDrift` — guarantees zero numerical drift for the annotated binding.
/// The compiler validates against the HW profile and emits a performance
/// advisory when the drift-free path is slower than the fast default.
#[derive(Debug, Clone, PartialEq)]
pub struct ZeroDriftAttr {
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundsAttr {
    pub min: Box<Expr>,
    pub max: Box<Expr>,
    pub span: Span,
}

