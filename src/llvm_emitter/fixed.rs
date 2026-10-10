//! Ordinary signed Q formats - `Q<int>.<frac>` outside `@ZeroDrift`.
//!
//! A value is held as the signed integer `value * 2^frac` in `i<int+frac>`
//! storage, and arithmetic stays in that domain. This backend used to give a
//! Q value `emit_type`'s `i32` default and treat it as an integer: `let x:
//! Q16.16 = 1.5` stored `fptosi 1.5` = 1, so `x > 1.0` was false.
//!
//! The semantics, chosen to fail closed rather than wrap:
//! * a float literal is quantised to the nearest representable value, ties
//!   away from zero - the rule `@ZeroDrift` uses (`emit_to_fixed`);
//! * `*` and `/` round their exact result the same way;
//! * a result outside the format's range, and a division by zero, trap
//!   (`llvm.trap`) instead of wrapping;
//! * a literal outside the range is a compile-time error.
//!
//! Only what the type checker admits is lowered: values of ONE Q format,
//! numeric literals (which take the Q type), `+ - * /`, unary `-` and the six
//! comparisons, through `let`, assignment, compound assignment, `return` and
//! a Y function's parameters. Everything else that would need the scale -
//! a struct field, a `match`, a built-in such as `print_int` - is refused by
//! name, never handed the raw integer.
use super::LlvmEmitter;
use crate::ast::{BinaryOp, Expr, UnaryOp};
use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct QFormat {
    pub bits: u32,
    pub frac: u32,
}

impl QFormat {
    /// `Q<int>.<frac>` with at least one integer (sign) bit. Any width parses,
    /// so that an unsupported one is refused by name rather than ignored.
    pub fn parse(name: &str) -> Option<Self> {
        let (integer, fraction) = name.strip_prefix('Q')?.split_once('.')?;
        let integer: u32 = integer.parse().ok()?;
        let frac: u32 = fraction.parse().ok()?;
        let bits = integer.checked_add(frac)?;
        (integer > 0).then_some(Self { bits, frac })
    }
    /// Storage this backend can pass through the C ABI unchanged.
    pub fn supported(self) -> bool {
        matches!(self.bits, 8 | 16 | 32 | 64)
    }
    pub fn llvm(self) -> String {
        format!("i{}", self.bits)
    }
    pub fn name(self) -> String {
        format!("Q{}.{}", self.bits - self.frac, self.frac)
    }
    fn min_raw(self) -> i128 {
        -(1i128 << (self.bits - 1))
    }
    fn max_raw(self) -> i128 {
        (1i128 << (self.bits - 1)) - 1
    }
}

fn is_comparison(op: &BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
    )
}

fn is_arithmetic(op: &BinaryOp) -> bool {
    matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod)
}

impl LlvmEmitter {
    /// The Q format an expression's VALUE has, if any. A `@ZeroDrift`
    /// accumulator is not an ordinary Q value: it has a representation of its
    /// own (`zero_drift.rs`).
    pub(super) fn q_format(&self, expr: &Expr) -> Option<QFormat> {
        match expr {
            Expr::Ident(name, _) if self.zero_drift.contains_key(name) => None,
            Expr::Ident(name, _) => self.locals_ast_type.get(name).and_then(|n| QFormat::parse(n)),
            Expr::Call { func, .. } => self
                .fn_ast_returns
                .get(&self.emit_call_target(func))
                .and_then(|n| QFormat::parse(n)),
            Expr::UnaryOp { op: UnaryOp::Neg, operand, .. } => self.q_format(operand),
            Expr::BinaryOp { left, op, right, .. } if is_arithmetic(op) => {
                self.q_format(left).or_else(|| self.q_format(right))
            }
            Expr::MemberAccess { .. } | Expr::Index { .. } | Expr::UnaryOp { op: UnaryOp::Deref, .. } => {
                QFormat::parse(self.infer_ast_type(expr).trim_start_matches('&').trim_start_matches("mut "))
            }
            _ => None,
        }
    }

    /// The Q format of an ordinary local, if it has one.
    pub(super) fn local_q_format(&self, name: &str) -> Option<QFormat> {
        if self.zero_drift.contains_key(name) {
            return None;
        }
        self.locals_ast_type.get(name).and_then(|n| QFormat::parse(n))
    }

    /// `emit_type`'s answer for a Q format: its storage, or a refusal by name.
    pub(super) fn q_storage(&mut self, fmt: QFormat) -> String {
        if !fmt.supported() {
            self.q_refuse(format!(
                "{} needs {}-bit storage; this backend holds Q formats of 8, 16, 32 or 64 bits",
                fmt.name(),
                fmt.bits
            ));
            return "i32".into();
        }
        fmt.llvm()
    }

    /// Lower a Q comparison or Q arithmetic; `None` for anything else, which
    /// `emit_expr` lowers as before.
    pub(super) fn emit_q_expr(&mut self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::BinaryOp { left, op, right, .. } if is_comparison(op) => {
                let fmt = self.q_format(left).or_else(|| self.q_format(right))?;
                Some(self.q_comparison(left, op, right, fmt))
            }
            Expr::BinaryOp { op, .. } if is_arithmetic(op) => {
                let fmt = self.q_format(expr)?;
                Some(self.q_raw(expr, fmt))
            }
            Expr::UnaryOp { op: UnaryOp::Neg, .. } => {
                let fmt = self.q_format(expr)?;
                Some(self.q_raw(expr, fmt))
            }
            _ => None,
        }
    }

    fn q_refuse(&mut self, message: String) -> String {
        self.emit_errors.push(format!("[LLVM host backend] {message}"));
        "0".into()
    }

    /// `expr` as raw storage of `dst`: the value times `2^frac`.
    pub(super) fn q_value(&mut self, expr: &Expr, dst: QFormat) -> String {
        if !dst.supported() {
            return self.q_refuse(format!(
                "{} needs {}-bit storage; this backend holds Q formats of 8, 16, 32 or 64 bits",
                dst.name(),
                dst.bits
            ));
        }
        if let Some(src) = self.q_format(expr) {
            if src != dst {
                return self.q_refuse(format!(
                    "a {} value where a {} is expected; Q formats do not convert implicitly",
                    src.name(),
                    dst.name()
                ));
            }
            return self.q_raw(expr, dst);
        }
        match expr {
            Expr::Ident(name, _) if self.zero_drift.contains_key(name) => self.q_refuse(format!(
                "the @ZeroDrift accumulator `{}` cannot be read as an ordinary {} value",
                crate::lexical_scope::source_name(name),
                dst.name()
            )),
            _ if Self::q_literal_raw(expr, dst).is_some() => self.q_constant(expr, dst),
            Expr::ZeroInit(_) => "0".into(),
            // A literal expression is evaluated in the destination's format,
            // exactly as the type checker typed it: each literal takes the Q
            // type, and so does each operation.
            Expr::BinaryOp { op, .. } if is_arithmetic(op) => self.q_raw(expr, dst),
            Expr::UnaryOp { op: UnaryOp::Neg, .. } => self.q_raw(expr, dst),
            _ => {
                let ty = self.infer_ast_type(expr);
                self.q_refuse(format!("cannot convert a {ty} value to {}", dst.name()))
            }
        }
    }

    /// A literal - possibly negated, `-8` being `-(8)` - as raw storage of
    /// `dst`, quantised at compile time: nearest, ties away from zero. `None`
    /// for anything that is not a literal. The negation is folded before the
    /// range check, so a format's most negative value is a valid literal.
    fn q_literal_raw(expr: &Expr, dst: QFormat) -> Option<Option<i128>> {
        match expr {
            Expr::IntLit(value, _) => Some((*value as i128).checked_mul(1i128 << dst.frac)),
            Expr::FloatLit(value, _) => {
                // `f64::round` rounds half away from zero.
                let rounded = (value * 2f64.powi(dst.frac as i32)).round();
                Some((rounded.is_finite() && rounded.abs() < 2f64.powi(100)).then(|| rounded as i128))
            }
            Expr::UnaryOp { op: UnaryOp::Neg, operand, .. } => {
                Self::q_literal_raw(operand, dst).map(|raw| raw.and_then(i128::checked_neg))
            }
            _ => None,
        }
    }

    fn q_constant(&mut self, expr: &Expr, dst: QFormat) -> String {
        match Self::q_literal_raw(expr, dst).flatten() {
            Some(raw) if raw >= dst.min_raw() && raw <= dst.max_raw() => raw.to_string(),
            _ => {
                let shown = Self::literal_text(expr);
                self.q_refuse(format!("{shown} is outside {}'s representable range", dst.name()))
            }
        }
    }

    fn literal_text(expr: &Expr) -> String {
        match expr {
            Expr::IntLit(v, _) => v.to_string(),
            Expr::FloatLit(v, _) => v.to_string(),
            Expr::UnaryOp { op: UnaryOp::Neg, operand, .. } => format!("-{}", Self::literal_text(operand)),
            _ => "literal".into(),
        }
    }

    fn q_raw(&mut self, expr: &Expr, fmt: QFormat) -> String {
        match expr {
            _ if Self::q_literal_raw(expr, fmt).is_some() => self.q_constant(expr, fmt),
            Expr::Ident(..) | Expr::Call { .. } | Expr::MemberAccess { .. } | Expr::Index { .. }
            | Expr::UnaryOp { op: UnaryOp::Deref, .. } => self.emit_expr(expr, None, None),
            Expr::UnaryOp { op: UnaryOp::Neg, operand, .. } => {
                let raw = self.q_value(operand, fmt);
                let wide = self.q_wide(&raw, fmt);
                let negated = self.q_bin("sub", Self::q_wide_ty(fmt), "0", &wide);
                self.q_narrow_checked(&negated, fmt)
            }
            Expr::BinaryOp { left, op, right, .. } if is_arithmetic(op) => {
                let l = self.q_value(left, fmt);
                let r = self.q_value(right, fmt);
                self.q_operation(op, &l, &r, fmt)
            }
            _ => self.q_refuse(format!("this expression has no {} lowering", fmt.name())),
        }
    }

    fn q_bin(&mut self, op: &str, ty: &str, left: &str, right: &str) -> String {
        let out = self.fresh_tmp();
        writeln!(self.output, "  {out} = {op} {ty} {left}, {right}").unwrap();
        out
    }

    fn q_select(&mut self, condition: &str, ty: &str, yes: &str, no: &str) -> String {
        let out = self.fresh_tmp();
        writeln!(self.output, "  {out} = select i1 {condition}, {ty} {yes}, {ty} {no}").unwrap();
        out
    }

    /// Exact intermediates: a product or a shifted dividend of two values of
    /// at most 32 bits stays below 2^62, of 64 bits below 2^126.
    fn q_wide_ty(fmt: QFormat) -> &'static str {
        if fmt.bits <= 32 { "i64" } else { "i128" }
    }

    fn q_wide(&mut self, raw: &str, fmt: QFormat) -> String {
        let out = self.fresh_tmp();
        writeln!(self.output, "  {out} = sext {} {raw} to {}", fmt.llvm(), Self::q_wide_ty(fmt)).unwrap();
        out
    }

    /// Branch to a trap unless `condition` holds.
    fn q_guard(&mut self, condition: &str) {
        if !self.called_functions.iter().any(|f| f == "llvm.trap") {
            self.called_functions.push("llvm.trap".into());
        }
        let good = self.fresh_label("q.ok");
        let bad = self.fresh_label("q.trap");
        writeln!(
            self.output,
            "  br i1 {condition}, label %{good}, label %{bad}\n{bad}:\n  call void @llvm.trap()\n  unreachable\n{good}:"
        )
        .unwrap();
    }

    /// Narrow an exact result to `fmt`'s storage, trapping when it does not fit.
    fn q_narrow_checked(&mut self, wide: &str, fmt: QFormat) -> String {
        let ty = Self::q_wide_ty(fmt);
        let lo = self.q_bin("icmp sge", ty, wide, &fmt.min_raw().to_string());
        let hi = self.q_bin("icmp sle", ty, wide, &fmt.max_raw().to_string());
        let fits = self.q_bin("and", "i1", &lo, &hi);
        self.q_guard(&fits);
        let out = self.fresh_tmp();
        writeln!(self.output, "  {out} = trunc {ty} {wide} to {}", fmt.llvm()).unwrap();
        out
    }

    /// The magnitude of `v` and whether `v` was negative. The magnitude of the
    /// most negative intermediate still fits: intermediates stay 2 bits short
    /// of the wide type.
    fn q_magnitude(&mut self, v: &str, ty: &str) -> (String, String) {
        let negative = self.q_bin("icmp slt", ty, v, "0");
        let flipped = self.q_bin("sub", ty, "0", v);
        (self.q_select(&negative, ty, &flipped, v), negative)
    }

    /// Reapply a sign to a magnitude.
    fn q_signed(&mut self, magnitude: &str, negative: &str, ty: &str) -> String {
        let flipped = self.q_bin("sub", ty, "0", magnitude);
        self.q_select(negative, ty, &flipped, magnitude)
    }

    /// `v / 2^frac`, rounded to nearest with ties away from zero.
    fn q_round_shift(&mut self, v: &str, frac: u32, ty: &str) -> String {
        if frac == 0 {
            return v.into();
        }
        let (a, negative) = self.q_magnitude(v, ty);
        let q = self.q_bin("lshr", ty, &a, &frac.to_string());
        let r = self.q_bin("and", ty, &a, &((1i128 << frac) - 1).to_string());
        let up = self.q_bin("icmp uge", ty, &r, &(1i128 << (frac - 1)).to_string());
        let bump = self.fresh_tmp();
        writeln!(self.output, "  {bump} = zext i1 {up} to {ty}").unwrap();
        let magnitude = self.q_bin("add", ty, &q, &bump);
        self.q_signed(&magnitude, &negative, ty)
    }

    /// `n / d` rounded to nearest with ties away from zero; `d` is non-zero.
    /// It works on magnitudes and rounds up when the remainder is at least
    /// half the divisor, tested as `r >= b - r` so that nothing is doubled.
    fn q_round_div(&mut self, n: &str, d: &str, ty: &str) -> String {
        let (a, n_negative) = self.q_magnitude(n, ty);
        let (b, d_negative) = self.q_magnitude(d, ty);
        let q = self.q_bin("udiv", ty, &a, &b);
        let r = self.q_bin("urem", ty, &a, &b);
        let rest = self.q_bin("sub", ty, &b, &r);
        let up = self.q_bin("icmp uge", ty, &r, &rest);
        let bump = self.fresh_tmp();
        writeln!(self.output, "  {bump} = zext i1 {up} to {ty}").unwrap();
        let magnitude = self.q_bin("add", ty, &q, &bump);
        let negative = self.q_bin("xor", "i1", &n_negative, &d_negative);
        self.q_signed(&magnitude, &negative, ty)
    }

    /// `l op r` on two raw values of `fmt`.
    pub(super) fn q_operation(&mut self, op: &BinaryOp, l: &str, r: &str, fmt: QFormat) -> String {
        let ty = Self::q_wide_ty(fmt);
        let l = self.q_wide(l, fmt);
        let r = self.q_wide(r, fmt);
        let exact = match op {
            BinaryOp::Add => self.q_bin("add", ty, &l, &r),
            BinaryOp::Sub => self.q_bin("sub", ty, &l, &r),
            BinaryOp::Mul => {
                let product = self.q_bin("mul", ty, &l, &r);
                self.q_round_shift(&product, fmt.frac, ty)
            }
            BinaryOp::Div => {
                let nonzero = self.q_bin("icmp ne", ty, &r, "0");
                self.q_guard(&nonzero);
                let numerator = self.q_bin("shl", ty, &l, &fmt.frac.to_string());
                self.q_round_div(&numerator, &r, ty)
            }
            _ => return self.q_refuse(format!("`{op:?}` is not defined on {}", fmt.name())),
        };
        self.q_narrow_checked(&exact, fmt)
    }

    fn q_comparison(&mut self, left: &Expr, op: &BinaryOp, right: &Expr, fmt: QFormat) -> String {
        let pred = match op {
            BinaryOp::Eq => "eq",
            BinaryOp::NotEq => "ne",
            BinaryOp::Lt => "slt",
            BinaryOp::Le => "sle",
            BinaryOp::Gt => "sgt",
            _ => "sge",
        };
        let l = self.q_value(left, fmt);
        let r = self.q_value(right, fmt);
        self.q_bin(&format!("icmp {pred}"), &fmt.llvm(), &l, &r)
    }
}
