#![no_main]
use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use y::ast::*;
use y::cpu_emitter::CpuEmitter;
use y::type_checker::TypeChecker;

#[derive(Debug)]
pub enum FuzzExpr {
    IntLit(i64),
    BoolLit(bool),
    Var(String),
    Add(Box<FuzzExpr>, Box<FuzzExpr>),
    Sub(Box<FuzzExpr>, Box<FuzzExpr>),
    Mul(Box<FuzzExpr>, Box<FuzzExpr>),
}

impl<'a> Arbitrary<'a> for FuzzExpr {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Self::arbitrary_depth(u, 0)
    }
}

impl FuzzExpr {
    fn arbitrary_depth(u: &mut Unstructured, depth: usize) -> arbitrary::Result<Self> {
        // Enforce maximum AST depth of 8 to avoid stack overflow in mutator
        if depth >= 8 {
            return match u.choose_index(3)? {
                0 => Ok(FuzzExpr::IntLit(u.arbitrary()?)),
                1 => Ok(FuzzExpr::BoolLit(u.arbitrary()?)),
                _ => Ok(FuzzExpr::Var(u.arbitrary()?)),
            };
        }

        let choice: u8 = u.int_in_range(0..=5)?;
        match choice {
            0 => Ok(FuzzExpr::IntLit(u.arbitrary()?)),
            1 => Ok(FuzzExpr::BoolLit(u.arbitrary()?)),
            2 => Ok(FuzzExpr::Var(u.arbitrary()?)),
            3 => Ok(FuzzExpr::Add(
                Box::new(Self::arbitrary_depth(u, depth + 1)?),
                Box::new(Self::arbitrary_depth(u, depth + 1)?),
            )),
            4 => Ok(FuzzExpr::Sub(
                Box::new(Self::arbitrary_depth(u, depth + 1)?),
                Box::new(Self::arbitrary_depth(u, depth + 1)?),
            )),
            _ => Ok(FuzzExpr::Mul(
                Box::new(Self::arbitrary_depth(u, depth + 1)?),
                Box::new(Self::arbitrary_depth(u, depth + 1)?),
            )),
        }
    }
}

fn build_ast_expr(fuzz: &FuzzExpr) -> Expr {
    let dummy_span = Span { line: 1, col: 1 };
    match fuzz {
        FuzzExpr::IntLit(n) => Expr::IntLit(*n, dummy_span),
        FuzzExpr::BoolLit(b) => Expr::BoolLit(*b, dummy_span),
        FuzzExpr::Var(name) => {
            let clean_name = if name.is_empty() { "x".to_string() } else { name.clone() };
            Expr::Ident(clean_name, dummy_span)
        }
        FuzzExpr::Add(l, r) => Expr::BinaryOp {
            op: BinaryOp::Add,
            left: Box::new(build_ast_expr(l)),
            right: Box::new(build_ast_expr(r)),
            span: dummy_span,
        },
        FuzzExpr::Sub(l, r) => Expr::BinaryOp {
            op: BinaryOp::Sub,
            left: Box::new(build_ast_expr(l)),
            right: Box::new(build_ast_expr(r)),
            span: dummy_span,
        },
        FuzzExpr::Mul(l, r) => Expr::BinaryOp {
            op: BinaryOp::Mul,
            left: Box::new(build_ast_expr(l)),
            right: Box::new(build_ast_expr(r)),
            span: dummy_span,
        },
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let fuzz_expr = match FuzzExpr::arbitrary(&mut u) {
        Ok(e) => e,
        Err(_) => return,
    };

    let dummy_span = Span { line: 1, col: 1 };
    let expr = build_ast_expr(&fuzz_expr);

    let const_decl = ConstDecl {
        name: "FUZZ_CONST".to_string(),
        ty: Type::Primitive("i32".to_string(), dummy_span.clone()),
        value: expr,
        span: dummy_span,
    };

    let program = Program {
        items: vec![Item::Const(const_decl)],
    };

    // 1. Semantic verification
    let mut tc = TypeChecker::new();
    tc.check_program(&program);

    // 2. CpuEmitter codegen pass verification
    let mut emitter = CpuEmitter::new();
    let _ = emitter.emit_program(&program);
});
