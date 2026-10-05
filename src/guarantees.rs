//! What the compiler checked, proved, or took on trust about each part of a
//! program - the facts `ydb verify` reports for a source line.
//!
//! The type checker records them as it checks (`TypeChecker::guarantees`),
//! `@require` adds what it evaluated, and the LLVM backend adds what it
//! substituted. Every fact says how it is established - a proof, a check, a
//! run-time check, or an assumption - and a proof that used an assumed range
//! names the assumption, because a proof from a false premise proves nothing.
//!
//! The facts travel two ways, through ONE serialiser so the two cannot
//! disagree: inside a `-g` binary (`Y_PROGRAM["guarantees"]` in the gdb
//! extension the program carries), and as a file (`--emit-guarantees`) for a
//! program that has no host binary to carry them.

use crate::ast::{BinaryOp, Expr, UnaryOp};

/// How a fact is established, strongest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    /// A machine-checked proof (z3, or the Rocq proofs) establishes it for
    /// this code.
    Proved,
    /// The compiler checked it, and would have refused the program otherwise.
    Checked,
    /// It is checked when the program runs: a violation stops the program.
    RunTime,
    /// Established by tests, not by a proof.
    Tested,
    /// Assumed. Nothing checks it.
    Trusted,
    /// Nothing checks it, by the program's own choice (`@unsafe`), or because
    /// the compiler cannot (a pointer whose length it does not know).
    NotChecked,
    /// The check was skipped (`Y_ALLOW_UNVERIFIED_INVARIANTS`).
    Unverified,
}

impl Status {
    pub fn name(self) -> &'static str {
        match self {
            Status::Proved => "proved",
            Status::Checked => "checked",
            Status::RunTime => "run-time",
            Status::Tested => "tested",
            Status::Trusted => "trusted",
            Status::NotChecked => "not-checked",
            Status::Unverified => "unverified",
        }
    }
}

/// Something a proof may rest on that nothing checks: a `@bounds` range taken
/// on trust, or CUDA's launch limits.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Assumption {
    /// The item it is written in (for its file); empty for one that has no
    /// place in the source.
    pub item: String,
    /// Its line, or 0 for one that has no place in the source.
    pub line: usize,
    pub what: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fact {
    /// The function or kernel it is in, spelled as `debug_info::item_names`
    /// spells it, which is what attributes it to a file.
    pub item: String,
    pub line: usize,
    pub col: usize,
    /// The last line it covers: a loop's invariant covers its body.
    pub end_line: usize,
    /// `safe`, `index`, `invariant`, `bounds`, `drift`, `require`, `gemm`.
    pub kind: &'static str,
    pub status: Status,
    /// What it is about: `v[i]`, `@invariant(i >= 0)`, `fn main`.
    pub what: String,
    pub detail: String,
    /// The assumptions a proof used, if any.
    pub rests_on: Vec<Assumption>,
}

/// A function or kernel, and the lines it spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemRange {
    pub name: String,
    /// `fn`, `kernel` or `method`.
    pub kind: &'static str,
    pub line: usize,
    pub end_line: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Guarantees {
    pub items: Vec<ItemRange>,
    pub facts: Vec<Fact>,
}

impl Guarantees {
    /// The `@bounds` taken on trust in `item` between `first` and `last`: what
    /// a claim about that code rests on when it uses their ranges.
    pub fn trusted_bounds(&self, item: &str, first: usize, last: usize) -> Vec<Assumption> {
        self.facts
            .iter()
            .filter(|f| f.kind == "bounds" && f.status == Status::Trusted && f.item == item)
            .filter(|f| f.line >= first && f.line <= last)
            .map(|f| Assumption { item: f.item.clone(), line: f.line, what: f.what.clone() })
            .collect()
    }

    /// The facts as JSON: `{"version": 1, "items": [...], "facts": [...]}`.
    /// `file_of` names the file an item was parsed from.
    ///
    /// The output is ASCII, so it can be embedded in a Python string literal
    /// as well as written to a file.
    pub fn to_json(&self, file_of: &dyn Fn(&str) -> String) -> String {
        let mut items: Vec<&ItemRange> = self.items.iter().collect();
        items.sort_by(|a, b| (file_of(&a.name), a.line, &a.name).cmp(&(file_of(&b.name), b.line, &b.name)));
        let mut facts: Vec<&Fact> = self.facts.iter().collect();
        facts.sort_by(|a, b| {
            (file_of(&a.item), a.line, a.col, a.end_line, a.kind, &a.what)
                .cmp(&(file_of(&b.item), b.line, b.col, b.end_line, b.kind, &b.what))
        });
        facts.dedup();
        let mut out = String::from("{\"version\": 1, \"items\": [");
        for (i, it) in items.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!(
                "{{\"name\": {}, \"kind\": {}, \"file\": {}, \"line\": {}, \"end\": {}}}",
                json_string(&it.name),
                json_string(it.kind),
                json_string(&file_of(&it.name)),
                it.line,
                it.end_line
            ));
        }
        out.push_str("], \"facts\": [");
        for (i, f) in facts.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let rests: Vec<String> = f
                .rests_on
                .iter()
                .map(|a| {
                    format!(
                        "{{\"file\": {}, \"line\": {}, \"what\": {}}}",
                        json_string(&if a.item.is_empty() && a.line == 0 { String::new() } else { file_of(&a.item) }),
                        a.line,
                        json_string(&a.what)
                    )
                })
                .collect();
            out.push_str(&format!(
                "{{\"item\": {}, \"file\": {}, \"line\": {}, \"col\": {}, \"end\": {}, \"kind\": {}, \"status\": {}, \"what\": {}, \"detail\": {}, \"rests_on\": [{}]}}",
                json_string(&f.item),
                json_string(&file_of(&f.item)),
                f.line,
                f.col,
                f.end_line,
                json_string(f.kind),
                json_string(f.status.name()),
                json_string(&f.what),
                json_string(&f.detail),
                rests.join(", ")
            ));
        }
        out.push_str("]}");
        out
    }
}

/// A JSON string literal in ASCII: everything outside printable ASCII is a
/// `\u` escape (a surrogate pair above the BMP), as JSON requires.
pub fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            ' '..='~' => out.push(c),
            _ => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
        }
    }
    out.push('"');
    out
}

/// An expression as the source spells it, for a fact's `what`. Display only:
/// a shape it does not render is `...`.
pub fn render(expr: &Expr) -> String {
    match expr {
        Expr::Ident(n, _) => n.clone(),
        Expr::IntLit(v, _) => v.to_string(),
        Expr::FloatLit(v, _) => v.to_string(),
        Expr::BoolLit(v, _) => v.to_string(),
        Expr::StringLit(v, _) => format!("{:?}", v),
        Expr::CharLit(v, _) => format!("{:?}", v),
        Expr::Index { base, index, .. } => format!("{}[{}]", render(base), render(index)),
        Expr::MemberAccess { base, member, .. } => format!("{}.{}", render(base), member),
        Expr::Path { namespace, member, .. } => format!("{}::{}", namespace, member),
        Expr::Call { func, args, .. } => {
            let a: Vec<String> = args.iter().map(render).collect();
            format!("{}({})", render(func), a.join(", "))
        }
        Expr::UnaryOp { op, operand, .. } => {
            let o = match op {
                UnaryOp::Neg => "-",
                UnaryOp::Not => "!",
                UnaryOp::Ref { mutable: true } => "&mut ",
                UnaryOp::Ref { mutable: false } => "&",
                UnaryOp::Deref => "*",
            };
            format!("{}{}", o, render_operand(operand))
        }
        Expr::BinaryOp { left, op, right, .. } => {
            format!("{} {} {}", render_operand(left), binary_op(op), render_operand(right))
        }
        _ => "...".to_string(),
    }
}

/// A binary operation as an operand is parenthesised, so `(a + b) * c` keeps
/// its meaning.
fn render_operand(expr: &Expr) -> String {
    match expr {
        Expr::BinaryOp { .. } => format!("({})", render(expr)),
        _ => render(expr),
    }
}

pub fn binary_op(op: &BinaryOp) -> &'static str {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_strings_are_ascii_and_escaped() {
        assert_eq!(json_string("a\"b\\c\nd"), "\"a\\\"b\\\\c\\nd\"");
        // A non-BMP character is a surrogate pair, as JSON requires.
        assert_eq!(json_string("\u{1F600}"), "\"\\ud83d\\ude00\"");
        assert_eq!(json_string("\u{e9}"), "\"\\u00e9\"");
    }

    #[test]
    fn identical_facts_are_written_once() {
        let f = Fact {
            item: "main".into(),
            line: 3,
            col: 5,
            end_line: 3,
            kind: "index",
            status: Status::Proved,
            what: "v[i]".into(),
            detail: "d".into(),
            rests_on: vec![],
        };
        let g = Guarantees { items: vec![], facts: vec![f.clone(), f] };
        let json = g.to_json(&|_| "p.ysu".to_string());
        assert_eq!(json.matches("\"v[i]\"").count(), 1, "{}", json);
    }
}
