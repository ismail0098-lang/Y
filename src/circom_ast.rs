// ============================================================
//  Y — circom front end: AST
//  circom_ast.rs
// ============================================================

#![allow(dead_code)]

use crate::zk_field::BigUint;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pos {
    pub line: usize,
    pub col: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalKind {
    Input,
    Output,
    Intermediate,
}

/// How a substitution relates its two sides.
///
/// The distinction between `<==` and `<--` is the single most safety-relevant
/// thing in the language: `<--` assigns a witness value and emits NO constraint,
/// so a circuit that uses it without a matching `===` is under-constrained and
/// the proof means nothing. Y must never silently turn one into the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignOp {
    /// `=` — plain variable assignment (compile-time / witness-time values).
    Var,
    /// `<==` or `==>` — assign and constrain.
    SignalConstrain,
    /// `<--` or `-->` — assign only, no constraint.
    SignalOnly,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    /// Field division (`/`): multiplication by the modular inverse.
    Div,
    /// Integer quotient (`\`).
    IntDiv,
    Mod,
    Pow,
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
    BitNot,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Number(BigUint, Pos),
    Var(String, Pos),
    /// `a[i]`, possibly chained.
    Index(Box<Expr>, Box<Expr>, Pos),
    /// `c.out` — a signal of an instantiated component.
    Member(Box<Expr>, String, Pos),
    Binary(BinOp, Box<Expr>, Box<Expr>, Pos),
    Unary(UnOp, Box<Expr>, Pos),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>, Pos),
    /// A call to a `function` (never to a template).
    Call(String, Vec<Expr>, Pos),
    /// `[a, b, c]` — an inline array, used for `var` initialisers.
    ArrayInline(Vec<Expr>, Pos),
    /// `T(args)` on the right of a `component` declaration.
    TemplateInst(String, Vec<Expr>, Pos),
    /// `T(targs)(in1, in2)` — a circom 2.1 anonymous component.
    ///
    /// Written in expression position, but circom does NOT let it appear
    /// inside arithmetic: `o <== One()(i) + 3` is refused with "This is the
    /// anonymous component whose use is not allowed". The only legal positions
    /// are the entire right-hand side of a substitution and an argument of
    /// another anonymous component, so the lowering handles it at those two
    /// sites and `eval_expr` refuses it everywhere else, matching circom.
    ///
    /// An input is `(None, expr)` when passed positionally and
    /// `(Some(name), expr)` for circom's `T()(a <== x)` form. circom requires
    /// the count to coincide with the template's inputs either way.
    AnonComp {
        template: String,
        targs: Vec<Expr>,
        inputs: Vec<(Option<String>, Expr)>,
        pos: Pos,
    },
    /// `(a, b)` on the left of a substitution — circom 2.1 tuple destructuring
    /// of an anonymous component's outputs. Legal nowhere else, and refused by
    /// name if it turns up in a value position.
    Tuple(Vec<Expr>, Pos),
}

impl Expr {
    pub fn pos(&self) -> Pos {
        match self {
            Expr::Number(_, p)
            | Expr::Var(_, p)
            | Expr::Index(_, _, p)
            | Expr::Member(_, _, p)
            | Expr::Binary(_, _, _, p)
            | Expr::Unary(_, _, p)
            | Expr::Ternary(_, _, _, p)
            | Expr::Call(_, _, p)
            | Expr::ArrayInline(_, p)
            | Expr::TemplateInst(_, _, p)
            | Expr::AnonComp { pos: p, .. }
            | Expr::Tuple(_, p) => *p,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Block(Vec<Stmt>, Pos),
    /// Several statements that are NOT a new scope.
    ///
    /// `var a, b;` and `signal x, y;` each parse to more than one declaration,
    /// and putting them in a `Block` made them declare into a scope that was
    /// popped on the way out - so `var a, b; a = 1;` reported "`a` is not a
    /// variable in scope". Signals happened to survive because `Frame::signals`
    /// is not scoped; vars did not.
    Seq(Vec<Stmt>, Pos),
    /// `signal input in[3];`
    DeclSignal {
        kind: SignalKind,
        name: String,
        dims: Vec<Expr>,
        /// circom allows `signal output out <== expr;`
        init: Option<(AssignOp, Expr)>,
        pos: Pos,
    },
    /// `var x[2] = [1, 2];`
    DeclVar {
        name: String,
        dims: Vec<Expr>,
        init: Option<Expr>,
        pos: Pos,
    },
    /// `component c[4];` or `component c = T(2);`
    DeclComponent {
        name: String,
        dims: Vec<Expr>,
        init: Option<Expr>,
        pos: Pos,
    },
    /// `lhs <== rhs`, `lhs <-- rhs`, `lhs = rhs`, `lhs += rhs`, ...
    Substitution {
        lhs: Expr,
        op: AssignOp,
        rhs: Expr,
        pos: Pos,
    },
    /// `a === b`
    ConstraintEq(Expr, Expr, Pos),
    If {
        cond: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
        pos: Pos,
    },
    For {
        init: Box<Stmt>,
        cond: Expr,
        step: Box<Stmt>,
        body: Box<Stmt>,
        pos: Pos,
    },
    While {
        cond: Expr,
        body: Box<Stmt>,
        pos: Pos,
    },
    Return(Expr, Pos),
    Assert(Expr, Pos),
    /// `log(...)` — accepted and discarded; it has no circuit meaning.
    Log(Vec<Expr>, Pos),
}

#[derive(Clone, Debug)]
pub struct Template {
    pub name: String,
    pub params: Vec<String>,
    pub body: Stmt,
    /// `template custom T(...)` — declares a PLONKish custom gate. Y has no
    /// PLONKish backend, so this is refused rather than compiled as if it were
    /// an ordinary template.
    pub is_custom: bool,
    pub pos: Pos,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub params: Vec<String>,
    pub body: Stmt,
    pub pos: Pos,
}

#[derive(Clone, Debug)]
pub struct MainComponent {
    pub template: String,
    pub args: Vec<Expr>,
    /// Names listed in `{public [a, b]}`. Everything else is private.
    pub public: Vec<String>,
    pub pos: Pos,
}

#[derive(Clone, Debug, Default)]
pub struct Program {
    pub templates: Vec<Template>,
    pub functions: Vec<Function>,
    pub main: Option<MainComponent>,
}

impl Program {
    pub fn template(&self, name: &str) -> Option<&Template> {
        self.templates.iter().find(|t| t.name == name)
    }
    pub fn function(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|f| f.name == name)
    }
}
