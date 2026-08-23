// ============================================================
//  Y — circom front end: parser
//  circom_parser.rs
// ============================================================
//
// Recursive descent over `circom_lexer::Token`, with `include` resolution.
//
// Anything outside the supported subset is REFUSED by name rather than skipped.
// A front end that quietly ignores a construct it does not understand emits a
// circuit that is missing constraints, and a circuit missing constraints still
// proves — it just proves something weaker than the author wrote. That failure
// is invisible in every artifact downstream.

#![allow(dead_code)]

use crate::circom_ast::*;
use crate::circom_lexer::{Lexer, Tok, Token};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

type PResult<T> = Result<T, String>;

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self {
        Parser { toks, pos: 0 }
    }

    /// Parse a file, resolving `include` against `search_paths`.
    ///
    /// Each file is parsed once; circomlib's headers are included many times
    /// over and re-parsing them is both slow and a duplicate-template error.
    pub fn parse_file(entry: &Path, search_paths: &[PathBuf]) -> PResult<Program> {
        let mut program = Program::default();
        let mut seen: HashSet<PathBuf> = HashSet::new();
        Self::parse_into(entry, search_paths, &mut program, &mut seen)?;
        Ok(program)
    }

    fn parse_into(
        path: &Path,
        search_paths: &[PathBuf],
        program: &mut Program,
        seen: &mut HashSet<PathBuf>,
    ) -> PResult<()> {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if !seen.insert(canonical.clone()) {
            return Ok(());
        }
        let src = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
        let toks = Lexer::new(&src)
            .tokenize()
            .map_err(|e| format!("{}: {}", path.display(), e))?;

        let mut p = Parser::new(toks);
        let includes = p
            .parse_program_into(program)
            .map_err(|e| format!("{}: {}", path.display(), e))?;

        let here = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        for inc in includes {
            let mut candidates = vec![here.join(&inc)];
            for sp in search_paths {
                candidates.push(sp.join(&inc));
                // circomlib is conventionally included as "circomlib/circuits/x.circom"
                // or bare "x.circom" from within circomlib itself.
                candidates.push(sp.join("circuits").join(&inc));
            }
            let found = candidates.iter().find(|c| c.is_file()).cloned();
            match found {
                Some(f) => Self::parse_into(&f, search_paths, program, seen)?,
                None => {
                    return Err(format!(
                        "{}: cannot resolve include {:?}; looked in {}",
                        path.display(),
                        inc,
                        candidates
                            .iter()
                            .map(|c| c.display().to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                }
            }
        }
        Ok(())
    }

    // ---- token helpers ----

    fn peek(&self) -> &Tok {
        &self.toks[self.pos.min(self.toks.len() - 1)].tok
    }

    fn peek_at(&self, n: usize) -> &Tok {
        &self.toks[(self.pos + n).min(self.toks.len() - 1)].tok
    }

    fn pos_of(&self) -> Pos {
        let t = &self.toks[self.pos.min(self.toks.len() - 1)];
        Pos { line: t.line, col: t.col }
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.pos.min(self.toks.len() - 1)].tok.clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == t {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: &Tok, what: &str) -> PResult<()> {
        if self.eat(t) {
            Ok(())
        } else {
            let p = self.pos_of();
            Err(format!("{}:{}: expected {}, found {:?}", p.line, p.col, what, self.peek()))
        }
    }

    fn ident(&mut self) -> PResult<String> {
        let p = self.pos_of();
        match self.bump() {
            Tok::Ident(s) => Ok(s),
            other => Err(format!("{}:{}: expected identifier, found {:?}", p.line, p.col, other)),
        }
    }

    // ---- top level ----

    /// Returns the list of `include` paths found in this file.
    pub fn parse_program_into(&mut self, program: &mut Program) -> PResult<Vec<String>> {
        let mut includes = Vec::new();
        loop {
            match self.peek().clone() {
                Tok::Eof => break,
                Tok::Pragma => {
                    self.bump();
                    // `pragma circom 2.0.0;` / `pragma custom_templates;`
                    while !matches!(self.peek(), Tok::Semi | Tok::Eof) {
                        self.bump();
                    }
                    self.expect(&Tok::Semi, "`;` after pragma")?;
                }
                Tok::Include => {
                    self.bump();
                    let p = self.pos_of();
                    match self.bump() {
                        Tok::Str(s) => includes.push(s),
                        other => {
                            return Err(format!(
                                "{}:{}: expected a quoted path after `include`, found {:?}",
                                p.line, p.col, other
                            ))
                        }
                    }
                    self.expect(&Tok::Semi, "`;` after include")?;
                }
                Tok::Template | Tok::Custom => {
                    let t = self.parse_template()?;
                    if program.template(&t.name).is_none() {
                        program.templates.push(t);
                    }
                }
                Tok::Function => {
                    let f = self.parse_function()?;
                    if program.function(&f.name).is_none() {
                        program.functions.push(f);
                    }
                }
                Tok::Component => {
                    let m = self.parse_main_component()?;
                    if program.main.is_some() {
                        return Err(format!(
                            "{}:{}: a second `component main` declaration",
                            m.pos.line, m.pos.col
                        ));
                    }
                    program.main = Some(m);
                }
                Tok::Bus => {
                    let p = self.pos_of();
                    return Err(format!(
                        "{}:{}: `bus` declarations are not supported by Y's circom front end. \
                         Buses are a circom 2.1.5+ feature for grouping signals; Y refuses rather \
                         than silently flattening one, which would change the signal layout your \
                         verifier expects.",
                        p.line, p.col
                    ));
                }
                other => {
                    let p = self.pos_of();
                    return Err(format!(
                        "{}:{}: expected `template`, `function`, `component main`, `include` or \
                         `pragma` at top level, found {:?}",
                        p.line, p.col, other
                    ));
                }
            }
        }
        Ok(includes)
    }

    fn parse_template(&mut self) -> PResult<Template> {
        let pos = self.pos_of();
        let is_custom = if self.eat(&Tok::Custom) {
            self.expect(&Tok::Template, "`template` after `custom`")?;
            true
        } else {
            self.expect(&Tok::Template, "`template`")?;
            // `template custom Name(...)` is the spelling circom actually uses.
            self.eat(&Tok::Custom)
        };
        let name = self.ident()?;
        let params = self.parse_param_list()?;
        let body = self.parse_block()?;
        Ok(Template { name, params, body, is_custom, pos })
    }

    fn parse_function(&mut self) -> PResult<Function> {
        let pos = self.pos_of();
        self.expect(&Tok::Function, "`function`")?;
        let name = self.ident()?;
        let params = self.parse_param_list()?;
        let body = self.parse_block()?;
        Ok(Function { name, params, body, pos })
    }

    fn parse_param_list(&mut self) -> PResult<Vec<String>> {
        let mut params = Vec::new();
        if self.eat(&Tok::LParen) {
            if !self.eat(&Tok::RParen) {
                loop {
                    params.push(self.ident()?);
                    if self.eat(&Tok::Comma) {
                        continue;
                    }
                    self.expect(&Tok::RParen, "`)` closing the parameter list")?;
                    break;
                }
            }
        }
        Ok(params)
    }

    fn parse_main_component(&mut self) -> PResult<MainComponent> {
        let pos = self.pos_of();
        self.expect(&Tok::Component, "`component`")?;
        let name = self.ident()?;
        if name != "main" {
            return Err(format!(
                "{}:{}: only `component main` may be declared at top level, found `component {}`",
                pos.line, pos.col, name
            ));
        }
        let mut public = Vec::new();
        if self.eat(&Tok::LBrace) {
            self.expect(&Tok::Public, "`public` inside `{...}`")?;
            self.expect(&Tok::LBracket, "`[` after `public`")?;
            if !self.eat(&Tok::RBracket) {
                loop {
                    public.push(self.ident()?);
                    if self.eat(&Tok::Comma) {
                        continue;
                    }
                    self.expect(&Tok::RBracket, "`]` closing the public list")?;
                    break;
                }
            }
            self.expect(&Tok::RBrace, "`}` closing the public declaration")?;
        }
        self.expect(&Tok::Assign, "`=` in `component main = T(...)`")?;
        let template = self.ident()?;
        let mut args = Vec::new();
        if self.eat(&Tok::LParen) {
            if !self.eat(&Tok::RParen) {
                loop {
                    args.push(self.parse_expr()?);
                    if self.eat(&Tok::Comma) {
                        continue;
                    }
                    self.expect(&Tok::RParen, "`)` closing the template arguments")?;
                    break;
                }
            }
        }
        self.expect(&Tok::Semi, "`;` after `component main`")?;
        Ok(MainComponent { template, args, public, pos })
    }

    // ---- statements ----

    fn parse_block(&mut self) -> PResult<Stmt> {
        let pos = self.pos_of();
        self.expect(&Tok::LBrace, "`{`")?;
        let mut stmts = Vec::new();
        while !self.eat(&Tok::RBrace) {
            if matches!(self.peek(), Tok::Eof) {
                return Err(format!("{}:{}: unterminated block", pos.line, pos.col));
            }
            stmts.push(self.parse_stmt()?);
        }
        Ok(Stmt::Block(stmts, pos))
    }

    fn parse_stmt(&mut self) -> PResult<Stmt> {
        let pos = self.pos_of();
        match self.peek().clone() {
            Tok::LBrace => self.parse_block(),
            Tok::Signal => self.parse_signal_decl(),
            Tok::Var => self.parse_var_decl(),
            Tok::Component => self.parse_component_decl(),
            Tok::If => {
                self.bump();
                self.expect(&Tok::LParen, "`(` after `if`")?;
                let cond = self.parse_expr()?;
                self.expect(&Tok::RParen, "`)` after the `if` condition")?;
                let then_branch = Box::new(self.parse_stmt()?);
                let else_branch = if self.eat(&Tok::Else) {
                    Some(Box::new(self.parse_stmt()?))
                } else {
                    None
                };
                Ok(Stmt::If { cond, then_branch, else_branch, pos })
            }
            Tok::For => {
                self.bump();
                self.expect(&Tok::LParen, "`(` after `for`")?;
                let init = Box::new(self.parse_simple_stmt(true)?);
                let cond = self.parse_expr()?;
                self.expect(&Tok::Semi, "`;` after the `for` condition")?;
                let step = Box::new(self.parse_simple_stmt(false)?);
                self.expect(&Tok::RParen, "`)` closing the `for` header")?;
                let body = Box::new(self.parse_stmt()?);
                Ok(Stmt::For { init, cond, step, body, pos })
            }
            Tok::While => {
                self.bump();
                self.expect(&Tok::LParen, "`(` after `while`")?;
                let cond = self.parse_expr()?;
                self.expect(&Tok::RParen, "`)` after the `while` condition")?;
                let body = Box::new(self.parse_stmt()?);
                Ok(Stmt::While { cond, body, pos })
            }
            Tok::Return => {
                self.bump();
                let e = self.parse_expr()?;
                self.expect(&Tok::Semi, "`;` after `return`")?;
                Ok(Stmt::Return(e, pos))
            }
            Tok::Assert => {
                self.bump();
                self.expect(&Tok::LParen, "`(` after `assert`")?;
                let e = self.parse_expr()?;
                self.expect(&Tok::RParen, "`)` after the assertion")?;
                self.expect(&Tok::Semi, "`;` after `assert`")?;
                Ok(Stmt::Assert(e, pos))
            }
            Tok::Log => {
                self.bump();
                self.expect(&Tok::LParen, "`(` after `log`")?;
                let mut args = Vec::new();
                if !self.eat(&Tok::RParen) {
                    loop {
                        // `log("text")` is legal; strings have no circuit meaning.
                        if let Tok::Str(_) = self.peek() {
                            self.bump();
                            args.push(Expr::Number(crate::zk_field::BigUint::zero(), pos));
                        } else {
                            args.push(self.parse_expr()?);
                        }
                        if self.eat(&Tok::Comma) {
                            continue;
                        }
                        self.expect(&Tok::RParen, "`)` closing `log`")?;
                        break;
                    }
                }
                self.expect(&Tok::Semi, "`;` after `log`")?;
                Ok(Stmt::Log(args, pos))
            }
            Tok::Semi => {
                self.bump();
                Ok(Stmt::Block(Vec::new(), pos))
            }
            _ => {
                let s = self.parse_simple_stmt(false)?;
                self.expect(&Tok::Semi, "`;` after statement")?;
                Ok(s)
            }
        }
    }

    /// A statement without its terminating `;`, for `for` headers and
    /// expression statements. `consume_semi` handles the `for` init clause,
    /// which owns its own `;`.
    fn parse_simple_stmt(&mut self, consume_semi: bool) -> PResult<Stmt> {
        let pos = self.pos_of();
        let s = if matches!(self.peek(), Tok::Var) {
            let d = self.parse_var_decl_no_semi()?;
            d
        } else {
            let lhs = self.parse_expr()?;
            // `RangeCheck(n)(a, b);` — an anonymous component as a whole
            // statement, legal in circom only when the template has no output
            // signals (otherwise: "This expression must be a tuple or an
            // anonymous component"). Encoded as a substitution into the EMPTY
            // tuple, so the output-count check in the lowerer is the same one
            // every other tuple goes through.
            if matches!(lhs, Expr::AnonComp { .. })
                && !matches!(
                    self.peek(),
                    Tok::AssignConstrainL | Tok::AssignL | Tok::Assign
                )
            {
                let s = Stmt::Substitution {
                    lhs: Expr::Tuple(Vec::new(), pos),
                    op: AssignOp::SignalConstrain,
                    rhs: lhs,
                    pos,
                };
                if consume_semi {
                    self.expect(&Tok::Semi, "`;` in the `for` header")?;
                }
                return Ok(s);
            }
            match self.peek().clone() {
                Tok::AssignConstrainL => {
                    self.bump();
                    let rhs = self.parse_expr()?;
                    Stmt::Substitution { lhs, op: AssignOp::SignalConstrain, rhs, pos }
                }
                Tok::AssignL => {
                    self.bump();
                    let rhs = self.parse_expr()?;
                    Stmt::Substitution { lhs, op: AssignOp::SignalOnly, rhs, pos }
                }
                // `expr ==> target` / `expr --> target`: the SOURCE is on the
                // left. Swapping these is how a reversed-arrow circuit silently
                // constrains the wrong wire.
                Tok::AssignConstrainR => {
                    self.bump();
                    let target = self.parse_expr()?;
                    Stmt::Substitution { lhs: target, op: AssignOp::SignalConstrain, rhs: lhs, pos }
                }
                Tok::AssignR => {
                    self.bump();
                    let target = self.parse_expr()?;
                    Stmt::Substitution { lhs: target, op: AssignOp::SignalOnly, rhs: lhs, pos }
                }
                Tok::ConstrainEq => {
                    self.bump();
                    let rhs = self.parse_expr()?;
                    Stmt::ConstraintEq(lhs, rhs, pos)
                }
                Tok::Assign => {
                    self.bump();
                    let rhs = self.parse_expr()?;
                    Stmt::Substitution { lhs, op: AssignOp::Var, rhs, pos }
                }
                op @ (Tok::PlusAssign
                | Tok::MinusAssign
                | Tok::StarAssign
                | Tok::SlashAssign
                | Tok::IntDivAssign
                | Tok::PercentAssign
                | Tok::PowAssign
                | Tok::ShlAssign
                | Tok::ShrAssign
                | Tok::AndAssign
                | Tok::OrAssign
                | Tok::XorAssign) => {
                    self.bump();
                    let rhs = self.parse_expr()?;
                    let bin = match op {
                        Tok::PlusAssign => BinOp::Add,
                        Tok::MinusAssign => BinOp::Sub,
                        Tok::StarAssign => BinOp::Mul,
                        Tok::SlashAssign => BinOp::Div,
                        Tok::IntDivAssign => BinOp::IntDiv,
                        Tok::PercentAssign => BinOp::Mod,
                        Tok::PowAssign => BinOp::Pow,
                        Tok::ShlAssign => BinOp::Shl,
                        Tok::ShrAssign => BinOp::Shr,
                        Tok::AndAssign => BinOp::BitAnd,
                        Tok::OrAssign => BinOp::BitOr,
                        Tok::XorAssign => BinOp::BitXor,
                        // Correct today only because the outer `op @ (..)`
                        // pattern admits exactly these eleven tokens, so the
                        // old `_ => BinOp::BitXor` WAS `XorAssign`. Add a
                        // twelfth to that list and forget this match, and the
                        // new operator silently becomes an XOR -- a wrong
                        // operator in a front end whose output is a circuit,
                        // which is the `Lt|Le|Gt|Ge => NotEq` bug in
                        // zk_emitter. Named explicitly so that is a panic
                        // rather than a proof of the wrong statement.
                        other => unreachable!(
                            "compound assignment {other:?} reached the operator                              table without a lowering; add it here as well as to                              the pattern above"
                        ),
                    };
                    Stmt::Substitution {
                        lhs: lhs.clone(),
                        op: AssignOp::Var,
                        rhs: Expr::Binary(bin, Box::new(lhs), Box::new(rhs), pos),
                        pos,
                    }
                }
                Tok::Inc | Tok::Dec => {
                    let is_inc = matches!(self.bump(), Tok::Inc);
                    let one = Expr::Number(crate::zk_field::BigUint::one(), pos);
                    Stmt::Substitution {
                        lhs: lhs.clone(),
                        op: AssignOp::Var,
                        rhs: Expr::Binary(
                            if is_inc { BinOp::Add } else { BinOp::Sub },
                            Box::new(lhs),
                            Box::new(one),
                            pos,
                        ),
                        pos,
                    }
                }
                other => {
                    return Err(format!(
                        "{}:{}: expected an assignment or constraint operator, found {:?}",
                        pos.line, pos.col, other
                    ))
                }
            }
        };
        if consume_semi {
            self.expect(&Tok::Semi, "`;` in the `for` header")?;
        }
        Ok(s)
    }

    fn parse_dims(&mut self) -> PResult<Vec<Expr>> {
        let mut dims = Vec::new();
        while self.eat(&Tok::LBracket) {
            dims.push(self.parse_expr()?);
            self.expect(&Tok::RBracket, "`]` closing an array dimension")?;
        }
        Ok(dims)
    }

    fn parse_signal_decl(&mut self) -> PResult<Stmt> {
        let pos = self.pos_of();
        self.expect(&Tok::Signal, "`signal`")?;
        let kind = if self.eat(&Tok::Input) {
            SignalKind::Input
        } else if self.eat(&Tok::Output) {
            SignalKind::Output
        } else {
            SignalKind::Intermediate
        };
        // `signal input {binary} x` — circom 2.1 tags. Refused: a tag is a
        // claimed property (e.g. "already range-checked") that other templates
        // are entitled to rely on, so ignoring one can drop a real constraint.
        if matches!(self.peek(), Tok::LBrace) {
            return Err(format!(
                "{}:{}: signal tags (`signal input {{...}} x`) are not supported by Y's circom \
                 front end. A tag is a claim other templates may rely on to skip checks, so \
                 ignoring it could drop a constraint that makes the circuit sound.",
                pos.line, pos.col
            ));
        }
        // `signal (a, b[n]) <== T()(x);` -- circom 2.1 declares a tuple of
        // signals and drives them in one statement. Exactly
        // `signal a; signal b[n]; (a, b) <== T()(x);`, and circom's own two
        // forms serialize to byte-identical `.r1cs`, so it is lowered as that
        // desugaring rather than given a meaning of its own.
        if matches!(self.peek(), Tok::LParen) {
            self.bump();
            let mut stmts = Vec::new();
            let mut targets = Vec::new();
            loop {
                let n = self.ident()?;
                let d = self.parse_dims()?;
                // `_` discards an output; there is no signal to declare for it.
                if n != "_" {
                    stmts.push(Stmt::DeclSignal {
                        kind,
                        name: n.clone(),
                        dims: d,
                        init: None,
                        pos,
                    });
                }
                targets.push(Expr::Var(n, pos));
                if self.eat(&Tok::Comma) {
                    continue;
                }
                self.expect(&Tok::RParen, "`)` closing the signal tuple")?;
                break;
            }
            let op = match self.peek() {
                Tok::AssignConstrainL => AssignOp::SignalConstrain,
                Tok::AssignL => AssignOp::SignalOnly,
                other => {
                    return Err(format!(
                        "{}:{}: a `signal (a, b)` declaration must be driven immediately, \
                         with `<==`; found {:?}",
                        pos.line, pos.col, other
                    ))
                }
            };
            self.bump();
            let rhs = self.parse_expr()?;
            self.expect(&Tok::Semi, "`;` after the signal declaration")?;
            stmts.push(Stmt::Substitution { lhs: Expr::Tuple(targets, pos), op, rhs, pos });
            return Ok(Stmt::Seq(stmts, pos));
        }
        let name = self.ident()?;
        let dims = self.parse_dims()?;
        let init = match self.peek().clone() {
            Tok::AssignConstrainL => {
                self.bump();
                Some((AssignOp::SignalConstrain, self.parse_expr()?))
            }
            Tok::AssignL => {
                self.bump();
                Some((AssignOp::SignalOnly, self.parse_expr()?))
            }
            _ => None,
        };
        // `signal a, b;` — several signals in one declaration.
        if matches!(self.peek(), Tok::Comma) {
            let mut stmts = vec![Stmt::DeclSignal {
                kind,
                name,
                dims: dims.clone(),
                init,
                pos,
            }];
            while self.eat(&Tok::Comma) {
                let n = self.ident()?;
                let d = self.parse_dims()?;
                stmts.push(Stmt::DeclSignal { kind, name: n, dims: d, init: None, pos });
            }
            self.expect(&Tok::Semi, "`;` after the signal declaration")?;
            return Ok(Stmt::Seq(stmts, pos));
        }
        self.expect(&Tok::Semi, "`;` after the signal declaration")?;
        Ok(Stmt::DeclSignal { kind, name, dims, init, pos })
    }

    fn parse_var_decl_no_semi(&mut self) -> PResult<Stmt> {
        let pos = self.pos_of();
        self.expect(&Tok::Var, "`var`")?;
        let name = self.ident()?;
        let dims = self.parse_dims()?;
        let init = if self.eat(&Tok::Assign) { Some(self.parse_expr()?) } else { None };
        Ok(Stmt::DeclVar { name, dims, init, pos })
    }

    fn parse_var_decl(&mut self) -> PResult<Stmt> {
        let pos = self.pos_of();
        let first = self.parse_var_decl_no_semi()?;
        if matches!(self.peek(), Tok::Comma) {
            let mut stmts = vec![first];
            while self.eat(&Tok::Comma) {
                let name = self.ident()?;
                let dims = self.parse_dims()?;
                let init = if self.eat(&Tok::Assign) { Some(self.parse_expr()?) } else { None };
                stmts.push(Stmt::DeclVar { name, dims, init, pos });
            }
            self.expect(&Tok::Semi, "`;` after the var declaration")?;
            return Ok(Stmt::Seq(stmts, pos));
        }
        self.expect(&Tok::Semi, "`;` after the var declaration")?;
        Ok(first)
    }

    fn parse_component_decl(&mut self) -> PResult<Stmt> {
        let pos = self.pos_of();
        self.expect(&Tok::Component, "`component`")?;
        let name = self.ident()?;
        let dims = self.parse_dims()?;
        let init = if self.eat(&Tok::Assign) {
            self.eat(&Tok::Parallel);
            let tpos = self.pos_of();
            let tname = self.ident()?;
            let mut args = Vec::new();
            if self.eat(&Tok::LParen) {
                if !self.eat(&Tok::RParen) {
                    loop {
                        args.push(self.parse_expr()?);
                        if self.eat(&Tok::Comma) {
                            continue;
                        }
                        self.expect(&Tok::RParen, "`)` closing the template arguments")?;
                        break;
                    }
                }
            }
            Some(Expr::TemplateInst(tname, args, tpos))
        } else {
            None
        };
        self.expect(&Tok::Semi, "`;` after the component declaration")?;
        Ok(Stmt::DeclComponent { name, dims, init, pos })
    }

    // ---- expressions (precedence climbing, circom's table) ----

    pub fn parse_expr(&mut self) -> PResult<Expr> {
        self.parse_ternary()
    }

    fn parse_ternary(&mut self) -> PResult<Expr> {
        let cond = self.parse_binary(0)?;
        if self.eat(&Tok::Question) {
            let pos = self.pos_of();
            let a = self.parse_expr()?;
            self.expect(&Tok::Colon, "`:` in a ternary")?;
            let b = self.parse_expr()?;
            return Ok(Expr::Ternary(Box::new(cond), Box::new(a), Box::new(b), pos));
        }
        Ok(cond)
    }

    fn bin_prec(t: &Tok) -> Option<(u8, BinOp)> {
        Some(match t {
            Tok::OrOr => (1, BinOp::Or),
            Tok::AndAnd => (2, BinOp::And),
            Tok::Pipe => (3, BinOp::BitOr),
            Tok::Caret => (4, BinOp::BitXor),
            Tok::Amp => (5, BinOp::BitAnd),
            Tok::Eq => (6, BinOp::Eq),
            Tok::Neq => (6, BinOp::Neq),
            Tok::Lt => (7, BinOp::Lt),
            Tok::Gt => (7, BinOp::Gt),
            Tok::Le => (7, BinOp::Le),
            Tok::Ge => (7, BinOp::Ge),
            Tok::Shl => (8, BinOp::Shl),
            Tok::Shr => (8, BinOp::Shr),
            Tok::Plus => (9, BinOp::Add),
            Tok::Minus => (9, BinOp::Sub),
            Tok::Star => (10, BinOp::Mul),
            Tok::Slash => (10, BinOp::Div),
            Tok::IntDiv => (10, BinOp::IntDiv),
            Tok::Percent => (10, BinOp::Mod),
            Tok::Pow => (11, BinOp::Pow),
            _ => return None,
        })
    }

    fn parse_binary(&mut self, min_prec: u8) -> PResult<Expr> {
        let mut lhs = self.parse_unary()?;
        loop {
            let Some((prec, op)) = Self::bin_prec(self.peek()) else {
                return Ok(lhs);
            };
            if prec < min_prec {
                return Ok(lhs);
            }
            let pos = self.pos_of();
            self.bump();
            // `**` is right-associative in circom; everything else is left.
            let next_min = if matches!(op, BinOp::Pow) { prec } else { prec + 1 };
            let rhs = self.parse_binary(next_min)?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs), pos);
        }
    }

    fn parse_unary(&mut self) -> PResult<Expr> {
        let pos = self.pos_of();
        match self.peek().clone() {
            Tok::Minus => {
                self.bump();
                Ok(Expr::Unary(UnOp::Neg, Box::new(self.parse_unary()?), pos))
            }
            Tok::Plus => {
                self.bump();
                self.parse_unary()
            }
            Tok::Not => {
                self.bump();
                Ok(Expr::Unary(UnOp::Not, Box::new(self.parse_unary()?), pos))
            }
            Tok::Tilde => {
                self.bump();
                Ok(Expr::Unary(UnOp::BitNot, Box::new(self.parse_unary()?), pos))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> PResult<Expr> {
        let mut e = self.parse_primary()?;
        loop {
            let pos = self.pos_of();
            if self.eat(&Tok::LBracket) {
                let idx = self.parse_expr()?;
                self.expect(&Tok::RBracket, "`]` closing an index")?;
                e = Expr::Index(Box::new(e), Box::new(idx), pos);
            } else if self.eat(&Tok::Dot) {
                // `1.5` reaches here as Number Dot Number. There are no floats
                // in a field, so say so rather than "expected identifier".
                if let Tok::Number(_) = self.peek() {
                    return Err(format!(
                        "{}:{}: floating-point literal; circom values are field elements",
                        pos.line, pos.col
                    ));
                }
                let field = self.ident()?;
                e = Expr::Member(Box::new(e), field, pos);
            } else {
                return Ok(e);
            }
        }
    }

    /// The input list of an anonymous component: `(a, b)` or
    /// `(name1 <== a, name2 <== b)`.
    ///
    /// circom accepts either form but not a mixture, and requires the count to
    /// match the template's inputs; both are checked in the lowerer, where the
    /// template's signals are known.
    fn parse_anon_inputs(&mut self) -> PResult<Vec<(Option<String>, Expr)>> {
        self.expect(&Tok::LParen, "`(` opening the anonymous component inputs")?;
        let mut inputs = Vec::new();
        if self.eat(&Tok::RParen) {
            return Ok(inputs);
        }
        loop {
            let named = matches!(self.peek(), Tok::Ident(_))
                && matches!(self.peek_at(1), Tok::AssignConstrainL);
            let name = if named {
                let n = self.ident()?;
                self.bump();
                Some(n)
            } else {
                None
            };
            inputs.push((name, self.parse_expr()?));
            if self.eat(&Tok::Comma) {
                continue;
            }
            self.expect(&Tok::RParen, "`)` closing the anonymous component inputs")?;
            break;
        }
        Ok(inputs)
    }

    fn parse_primary(&mut self) -> PResult<Expr> {
        let pos = self.pos_of();
        match self.bump() {
            Tok::Number(n) => Ok(Expr::Number(n, pos)),
            Tok::Ident(name) => {
                if self.eat(&Tok::LParen) {
                    let mut args = Vec::new();
                    if !self.eat(&Tok::RParen) {
                        loop {
                            args.push(self.parse_expr()?);
                            if self.eat(&Tok::Comma) {
                                continue;
                            }
                            self.expect(&Tok::RParen, "`)` closing the call arguments")?;
                            break;
                        }
                    }
                    // `T(targs)(inputs)` — a circom 2.1 anonymous component.
                    // A second argument list can follow nothing else in the
                    // grammar (circom has no first-class functions), so this is
                    // unambiguous.
                    if matches!(self.peek(), Tok::LParen) {
                        let inputs = self.parse_anon_inputs()?;
                        return Ok(Expr::AnonComp { template: name, targs: args, inputs, pos });
                    }
                    Ok(Expr::Call(name, args, pos))
                } else {
                    Ok(Expr::Var(name, pos))
                }
            }
            Tok::LParen => {
                let e = self.parse_expr()?;
                // `(a, b) <== T()(x)` — a tuple, legal only on the left of a
                // substitution. Parsed here because `(a)` and `(a, b)` share a
                // prefix; a tuple reaching a value position is refused by the
                // lowerer, by name.
                if matches!(self.peek(), Tok::Comma) {
                    let mut items = vec![e];
                    while self.eat(&Tok::Comma) {
                        items.push(self.parse_expr()?);
                    }
                    self.expect(&Tok::RParen, "`)` closing a tuple")?;
                    return Ok(Expr::Tuple(items, pos));
                }
                self.expect(&Tok::RParen, "`)`")?;
                Ok(e)
            }
            Tok::LBracket => {
                let mut items = Vec::new();
                if !self.eat(&Tok::RBracket) {
                    loop {
                        items.push(self.parse_expr()?);
                        if self.eat(&Tok::Comma) {
                            continue;
                        }
                        self.expect(&Tok::RBracket, "`]` closing an array literal")?;
                        break;
                    }
                }
                Ok(Expr::ArrayInline(items, pos))
            }
            other => Err(format!(
                "{}:{}: expected an expression, found {:?}",
                pos.line, pos.col, other
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> PResult<Program> {
        let toks = Lexer::new(src).tokenize()?;
        let mut p = Parser::new(toks);
        let mut prog = Program::default();
        p.parse_program_into(&mut prog)?;
        Ok(prog)
    }

    #[test]
    fn parses_multiplier() {
        let prog = parse(
            "pragma circom 2.0.0;\n\
             template Multiplier2() {\n\
                 signal input a;\n\
                 signal input b;\n\
                 signal output c;\n\
                 c <== a * b;\n\
             }\n\
             component main = Multiplier2();",
        )
        .unwrap();
        assert_eq!(prog.templates.len(), 1);
        assert_eq!(prog.main.as_ref().unwrap().template, "Multiplier2");
    }

    /// `a ==> b` means "assign a INTO b". Parsing it as `b <== a` with the
    /// operands the wrong way round constrains the wrong signal and still
    /// compiles.
    #[test]
    fn reversed_arrows_put_the_source_on_the_left() {
        let prog = parse("template T() { signal input a; signal output b; a ==> b; }").unwrap();
        let Stmt::Block(stmts, _) = &prog.templates[0].body else { panic!() };
        let Stmt::Substitution { lhs, rhs, op, .. } = &stmts[2] else {
            panic!("expected a substitution, got {:?}", stmts[2])
        };
        assert_eq!(*op, AssignOp::SignalConstrain);
        assert!(matches!(lhs, Expr::Var(n, _) if n == "b"), "target must be b");
        assert!(matches!(rhs, Expr::Var(n, _) if n == "a"), "source must be a");
    }

    #[test]
    fn parses_public_main_and_args() {
        let prog =
            parse("template T(n) { signal input x; } component main {public [x]} = T(4);").unwrap();
        let m = prog.main.unwrap();
        assert_eq!(m.public, vec!["x".to_string()]);
        assert_eq!(m.args.len(), 1);
    }

    #[test]
    fn power_is_right_associative() {
        let prog = parse("function f() { return 2 ** 3 ** 2; }").unwrap();
        let Stmt::Block(stmts, _) = &prog.functions[0].body else { panic!() };
        let Stmt::Return(Expr::Binary(BinOp::Pow, _, rhs, _), _) = &stmts[0] else {
            panic!("expected a power expression")
        };
        assert!(
            matches!(**rhs, Expr::Binary(BinOp::Pow, _, _, _)),
            "2 ** 3 ** 2 must group as 2 ** (3 ** 2)"
        );
    }

    /// Unsupported constructs must be named, not skipped.
    #[test]
    fn unsupported_constructs_are_refused_by_name() {
        let bus = parse("bus Point { signal x; }").unwrap_err();
        assert!(bus.contains("bus"), "{}", bus);

        let tag = parse("template T() { signal input {binary} x; }").unwrap_err();
        assert!(tag.contains("tag"), "{}", tag);
    }
}
