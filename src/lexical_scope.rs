// ============================================================
//  Y — lexical scoping for a backend that keeps one slot per name
//  lexical_scope.rs
// ============================================================
//
//! Every binding a function makes gets a name of its own.
//!
//! Y is lexically scoped. A `let` is visible from the statement after it to
//! the end of its block, a `for` loop's variable inside the loop, a `match`
//! arm's bindings inside the arm; an inner binding of a name SHADOWS an outer
//! one for its scope rather than replacing it. The type checker resolves
//! names that way (a scope per block, per loop and per arm).
//!
//! The LLVM backend keeps one stack slot per NAME per function, so before
//! this its programs did not follow those rules:
//!
//! * `let a = 1; @safe { let a = 2; } return a;` returned 2, where the
//!   language says 1 - the inner `let` wrote the outer binding's slot.
//! * `let x: I32 = 1; let x: F64 = 2.5;` reused the `i32` slot for the `F64`,
//!   so the second `x` held `fptosi 2.5`.
//! * Two `for i` loops in one function emitted two `%i` allocas, which
//!   `clang` rejects ("multiple definition of local value named 'i'").
//!
//! [`unique_bindings`] fixes all three before the backend sees the function,
//! by renaming: the first binding of a name keeps it, and every later
//! binding of the same name in the function becomes `name.1`, `name.2`, ...
//! No Y identifier contains a `.`, so a renamed binding cannot collide with a
//! name in the source, and [`source_name`] recovers the source spelling for
//! the debugger and for diagnostics. Every USE of a name is rewritten to the
//! binding it resolves to under the type checker's rules.
//!
//! A function that binds no name twice comes back unchanged, so its emitted
//! module is byte-for-byte what it was.
//!
//! The walk is exhaustive - no `_ =>` arm over `Stmt`, `Expr`, `Type` or
//! `MatchPattern` - so a variant added later is a compile error here rather
//! than a place where a use silently keeps resolving to the wrong binding.

use crate::ast::*;
use std::collections::{HashMap, HashSet};

/// The name a binding has in the source: a renamed one is `name.N`.
pub fn source_name(name: &str) -> &str {
    match name.find('.') {
        Some(dot) => &name[..dot],
        None => name,
    }
}

/// `body` with every binding renamed apart, as described in the module doc.
/// `params` are bound first and keep their names.
pub fn unique_bindings(params: &[Param], body: &Block) -> Block {
    let mut r = Renamer::default();
    r.push();
    for p in params {
        r.declare(&p.name);
    }
    let mut body = body.clone();
    r.block(&mut body);
    r.pop();
    body
}

#[derive(Default)]
struct Renamer {
    /// Every name a binding of this function has been given.
    taken: HashSet<String>,
    /// Innermost last: source name -> the name its binding was given.
    scopes: Vec<HashMap<String, String>>,
}

impl Renamer {
    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop(&mut self) {
        self.scopes.pop();
    }

    /// Bind `name` in the innermost scope; returns the name it is given.
    fn declare(&mut self, name: &str) -> String {
        let given = if self.taken.insert(name.to_string()) {
            name.to_string()
        } else {
            let mut n = 1;
            loop {
                let candidate = format!("{}.{}", name, n);
                if self.taken.insert(candidate.clone()) {
                    break candidate;
                }
                n += 1;
            }
        };
        self.scopes
            .last_mut()
            .expect("a binding is always made inside a scope")
            .insert(name.to_string(), given.clone());
        given
    }

    /// Rewrite a use of `name` to the binding it resolves to. A name bound
    /// nowhere in the function - a global, a function, an enum variant - is
    /// left as it is.
    fn use_name(&self, name: &mut String) {
        if let Some(given) = self.scopes.iter().rev().find_map(|s| s.get(name.as_str())) {
            *name = given.clone();
        }
    }

    fn block(&mut self, b: &mut Block) {
        self.push();
        for s in &mut b.stmts {
            self.stmt(s);
        }
        self.pop();
    }

    fn stmt(&mut self, s: &mut Stmt) {
        match s {
            Stmt::Let { name, ty, init, cache_policy: _, zero_drift: _, bounds, span: _ } => {
                // The type, the initialiser and `@bounds` are all evaluated
                // before the new binding exists: `let x = x + 1;` reads the
                // OUTER `x`.
                if let Some(t) = ty {
                    self.ty(t);
                }
                if let Some(e) = init {
                    self.expr(e);
                }
                if let Some(b) = bounds {
                    self.expr(&mut b.min);
                    self.expr(&mut b.max);
                }
                *name = self.declare(name);
            }
            Stmt::TypeAlias { name: _, ty, span: _ } => self.ty(ty),
            Stmt::For {
                loop_var,
                start,
                end,
                step,
                body,
                invariant,
                is_uniform_branch: _,
                tile,
                prefetch_stride,
                span: _,
            } => {
                // The bounds and the step are checked in the enclosing scope.
                self.expr(start);
                self.expr(end);
                if let Some(st) = step {
                    self.expr(st);
                }
                if let Some(t) = tile {
                    self.expr(&mut t.block_m);
                    self.expr(&mut t.block_n);
                    if let Some(k) = &mut t.block_k {
                        self.expr(k);
                    }
                }
                if let Some(p) = prefetch_stride {
                    if let Some(st) = &mut p.stride {
                        self.expr(st);
                    }
                }
                self.push();
                *loop_var = self.declare(loop_var);
                if let Some(inv) = invariant {
                    self.expr(inv);
                }
                self.block(body);
                self.pop();
            }
            Stmt::Assign { target, value, span: _ } => {
                self.expr(target);
                self.expr(value);
            }
            Stmt::Expr(e) => self.expr(e),
            Stmt::Return(e, _) => {
                if let Some(e) = e {
                    self.expr(e);
                }
            }
            Stmt::Chisel(b, _) => self.block(b),
            Stmt::If { condition, then_block, else_block, is_uniform_branch: _, span: _ } => {
                self.expr(condition);
                self.block(then_block);
                if let Some(eb) = else_block {
                    self.block(eb);
                }
            }
            Stmt::While {
                condition,
                body,
                invariant,
                max_iterations: _,
                is_uniform_branch: _,
                span: _,
            } => {
                self.expr(condition);
                if let Some(inv) = invariant {
                    self.expr(inv);
                }
                self.block(body);
            }
            Stmt::Break { span: _ } => {}
            Stmt::Match { scrutinee, arms, span: _ } => {
                self.expr(scrutinee);
                for arm in arms {
                    self.push();
                    match &mut arm.pattern {
                        // The type checker binds a bare name to the scrutinee.
                        MatchPattern::Ident(name, _) => *name = self.declare(name),
                        MatchPattern::EnumVariant { path: _, variant: _, bindings, span: _ } => {
                            for b in bindings {
                                *b = self.declare(b);
                            }
                        }
                        MatchPattern::Literal(e) => self.expr(e),
                        MatchPattern::Wildcard(_) => {}
                    }
                    self.expr(&mut arm.body);
                    self.pop();
                }
            }
            Stmt::CompoundAssign { target, op: _, value, span: _ } => {
                self.expr(target);
                self.expr(value);
            }
            Stmt::SafeBlock(b, _) => self.block(b),
            Stmt::GhostBlock(b, _) => self.block(b),
            Stmt::ClockDomainBlock { clock, body, span: _ } => {
                self.expr(clock);
                self.block(body);
            }
            Stmt::CompileTimeAssert { condition, message: _, span: _ } => self.expr(condition),
            Stmt::HintBlock { outputs, body, span: _ } => {
                for o in outputs {
                    self.use_name(o);
                }
                self.block(body);
            }
        }
    }

    fn expr(&mut self, e: &mut Expr) {
        match e {
            Expr::Ident(name, _) => self.use_name(name),
            Expr::IntLit(..)
            | Expr::FloatLit(..)
            | Expr::StringLit(..)
            | Expr::CharLit(..)
            | Expr::BoolLit(..)
            | Expr::SelfLit(_)
            | Expr::ZeroInit(_)
            | Expr::Path { .. } => {}
            Expr::Call { func, args, span: _ } => {
                self.callee(func);
                for a in args {
                    self.expr(a);
                }
            }
            Expr::GenericCall { func, generic_args, args, span: _ } => {
                self.callee(func);
                for t in generic_args {
                    self.ty(t);
                }
                for a in args {
                    self.expr(a);
                }
            }
            Expr::Index { base, index, span: _ } => {
                self.expr(base);
                self.expr(index);
            }
            Expr::MemberAccess { base, member: _, span: _ } => self.expr(base),
            Expr::BinaryOp { left, op: _, right, span: _ } => {
                self.expr(left);
                self.expr(right);
            }
            Expr::UnaryOp { op: _, operand, span: _ } => self.expr(operand),
            Expr::BlockExpr(b, _) => self.block(b),
            Expr::StructLit { name: _, fields, span: _ } => {
                for (_, v) in fields {
                    self.expr(v);
                }
            }
        }
    }

    /// A bare name in call position is a FUNCTION: the type checker refuses a
    /// call to a local value, so it can never be a binding of this function.
    /// Anything else (`obj.method(..)`) is an expression like any other.
    fn callee(&mut self, func: &mut Expr) {
        if !matches!(func, Expr::Ident(..)) {
            self.expr(func);
        }
    }

    fn ty(&mut self, t: &mut Type) {
        match t {
            Type::Primitive(..) | Type::Ident(..) => {}
            Type::Generic { base: _, args, span: _ } => {
                for a in args {
                    match a {
                        GenericArg::Type(t) => self.ty(t),
                        GenericArg::Value(e) => self.expr(e),
                        GenericArg::Named { name: _, val } => self.expr(val),
                    }
                }
            }
            Type::Array { element, size, span: _ } => {
                self.ty(element);
                self.expr(size);
            }
            Type::Reference { mutable: _, inner, span: _ } => self.ty(inner),
            Type::BlockTile { element, size, span: _ } => {
                self.ty(element);
                self.expr(size);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span() -> Span {
        Span { line: 1, col: 1 }
    }
    fn ident(n: &str) -> Expr {
        Expr::Ident(n.into(), span())
    }
    fn int(v: i64) -> Expr {
        Expr::IntLit(v, span())
    }
    fn let_(n: &str, init: Expr) -> Stmt {
        Stmt::Let {
            name: n.into(),
            ty: None,
            init: Some(init),
            cache_policy: None,
            zero_drift: None,
            bounds: None,
            span: span(),
        }
    }
    fn block(stmts: Vec<Stmt>) -> Block {
        Block { stmts, span: span() }
    }
    fn names_bound(b: &Block) -> Vec<String> {
        b.stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::Let { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_function_that_binds_no_name_twice_is_unchanged() {
        let body = block(vec![let_("a", int(1)), let_("b", ident("a")), Stmt::Return(Some(ident("b")), span())]);
        assert_eq!(unique_bindings(&[], &body), body);
    }

    #[test]
    fn an_inner_binding_shadows_and_the_outer_one_comes_back() {
        // let a = 1; @safe { let a = 2; let u = a; } return a;
        let body = block(vec![
            let_("a", int(1)),
            Stmt::SafeBlock(block(vec![let_("a", int(2)), let_("u", ident("a"))]), span()),
            Stmt::Return(Some(ident("a")), span()),
        ]);
        let out = unique_bindings(&[], &body);
        let Stmt::SafeBlock(inner, _) = &out.stmts[1] else { panic!() };
        assert_eq!(names_bound(inner), vec!["a.1", "u"]);
        assert_eq!(inner.stmts[1], let_("u", ident("a.1")));
        assert_eq!(out.stmts[2], Stmt::Return(Some(ident("a")), span()));
    }

    #[test]
    fn an_initialiser_reads_the_binding_it_shadows() {
        // let x = 1; let x = x + 1; return x;
        let plus = Expr::BinaryOp {
            left: Box::new(ident("x")),
            op: BinaryOp::Add,
            right: Box::new(int(1)),
            span: span(),
        };
        let body = block(vec![let_("x", int(1)), let_("x", plus), Stmt::Return(Some(ident("x")), span())]);
        let out = unique_bindings(&[], &body);
        assert_eq!(names_bound(&out), vec!["x", "x.1"]);
        let Stmt::Let { init: Some(Expr::BinaryOp { left, .. }), .. } = &out.stmts[1] else { panic!() };
        assert_eq!(**left, ident("x"));
        assert_eq!(out.stmts[2], Stmt::Return(Some(ident("x.1")), span()));
    }

    #[test]
    fn a_parameter_keeps_its_name_and_a_let_of_it_does_not() {
        let p = Param { name: "n".into(), ty: Type::Primitive("I32".into(), span()), span: span() };
        let body = block(vec![let_("n", ident("n")), Stmt::Return(Some(ident("n")), span())]);
        let out = unique_bindings(&[p], &body);
        assert_eq!(out.stmts[0], let_("n.1", ident("n")));
        assert_eq!(out.stmts[1], Stmt::Return(Some(ident("n.1")), span()));
    }

    #[test]
    fn a_callee_is_never_renamed() {
        // let f = 1; let g = f(f);  - `f` the function, `f` the local
        let call = Expr::Call { func: Box::new(ident("f")), args: vec![ident("f")], span: span() };
        let body = block(vec![let_("f", int(1)), let_("f", int(2)), let_("g", call)]);
        let out = unique_bindings(&[], &body);
        let Stmt::Let { init: Some(Expr::Call { func, args, .. }), .. } = &out.stmts[2] else { panic!() };
        assert_eq!(**func, ident("f"));
        assert_eq!(args[0], ident("f.1"));
    }

    #[test]
    fn the_source_name_is_recovered() {
        assert_eq!(source_name("x"), "x");
        assert_eq!(source_name("x.1"), "x");
        assert_eq!(source_name("x.12"), "x");
    }
}
