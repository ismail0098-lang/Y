// ============================================================
//  Y — circom front end: lexer
//  circom_lexer.rs
// ============================================================
//
// Y's advantage is a compiler back end, and a back end nobody can reach is not
// a product. No team rewrites an audited circuit in a new language for a build
// speed win, so the front end has to be the language they already have.
//
// This tokenises circom 2.x. It is deliberately a separate lexer from Y's own:
// the two languages share almost no lexical structure (`<==`, `<--`, `===`,
// `\` as integer division against `/` as FIELD division) and folding them
// together would make both harder to reason about.


use crate::zk_field::BigUint;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tok {
    // literals & names
    Ident(String),
    Number(BigUint),
    Str(String),

    // keywords
    Pragma,
    Include,
    Template,
    Function,
    Component,
    Signal,
    Input,
    Output,
    Var,
    For,
    While,
    If,
    Else,
    Return,
    Assert,
    Log,
    Public,
    Custom,
    Parallel,
    Bus,

    // signal operators — the ones that make circom circom
    /// `<==`  assign and constrain
    AssignConstrainL,
    /// `==>`  assign and constrain, reversed
    AssignConstrainR,
    /// `<--`  assign WITHOUT constraining
    AssignL,
    /// `-->`  assign without constraining, reversed
    AssignR,
    /// `===`  constrain only
    ConstrainEq,

    // assignment
    Assign,
    PlusAssign,
    MinusAssign,
    StarAssign,
    SlashAssign,
    /// `\=` — compound integer division. circom's `\` is quotient, distinct
    /// from `/`, which is field division by the modular inverse.
    IntDivAssign,
    PercentAssign,
    PowAssign,
    ShlAssign,
    ShrAssign,
    AndAssign,
    OrAssign,
    XorAssign,
    Inc,
    Dec,

    // arithmetic
    Plus,
    Minus,
    Star,
    Pow,
    /// `/` — FIELD division in circom, not integer division.
    Slash,
    /// `\` — integer (quotient) division.
    IntDiv,
    Percent,

    // comparison / logic
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    AndAnd,
    OrOr,
    Not,

    // bitwise
    Amp,
    Pipe,
    Caret,
    Tilde,
    Shl,
    Shr,

    // punctuation
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Semi,
    Dot,
    Question,
    Colon,

    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub line: usize,
    pub col: usize,
}

pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: usize,
    col: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Lexer { src: src.as_bytes(), pos: 0, line: 1, col: 1 }
    }

    fn peek(&self) -> u8 {
        *self.src.get(self.pos).unwrap_or(&0)
    }

    fn peek_at(&self, n: usize) -> u8 {
        *self.src.get(self.pos + n).unwrap_or(&0)
    }

    fn bump(&mut self) -> u8 {
        let c = self.peek();
        self.pos += 1;
        if c == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        c
    }

    fn skip_trivia(&mut self) -> Result<(), String> {
        loop {
            let c = self.peek();
            if c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
                self.bump();
            } else if c == b'/' && self.peek_at(1) == b'/' {
                while self.peek() != b'\n' && self.pos < self.src.len() {
                    self.bump();
                }
            } else if c == b'/' && self.peek_at(1) == b'*' {
                let (l, c0) = (self.line, self.col);
                self.bump();
                self.bump();
                loop {
                    if self.pos >= self.src.len() {
                        return Err(format!("{}:{}: unterminated block comment", l, c0));
                    }
                    if self.peek() == b'*' && self.peek_at(1) == b'/' {
                        self.bump();
                        self.bump();
                        break;
                    }
                    self.bump();
                }
            } else {
                return Ok(());
            }
        }
    }

    pub fn tokenize(mut self) -> Result<Vec<Token>, String> {
        let mut out = Vec::new();
        loop {
            self.skip_trivia()?;
            let (line, col) = (self.line, self.col);
            if self.pos >= self.src.len() {
                out.push(Token { tok: Tok::Eof, line, col });
                return Ok(out);
            }
            let tok = self.next_token()?;
            out.push(Token { tok, line, col });
        }
    }

    fn next_token(&mut self) -> Result<Tok, String> {
        let c = self.peek();
        let (line, col) = (self.line, self.col);

        if c.is_ascii_alphabetic() || c == b'_' {
            let start = self.pos;
            while self.peek().is_ascii_alphanumeric() || self.peek() == b'_' {
                self.bump();
            }
            let word = std::str::from_utf8(&self.src[start..self.pos]).unwrap().to_string();
            return Ok(match word.as_str() {
                "pragma" => Tok::Pragma,
                "include" => Tok::Include,
                "template" => Tok::Template,
                "function" => Tok::Function,
                "component" => Tok::Component,
                "signal" => Tok::Signal,
                "input" => Tok::Input,
                "output" => Tok::Output,
                "var" => Tok::Var,
                "for" => Tok::For,
                "while" => Tok::While,
                "if" => Tok::If,
                "else" => Tok::Else,
                "return" => Tok::Return,
                "assert" => Tok::Assert,
                "log" => Tok::Log,
                "public" => Tok::Public,
                "custom" => Tok::Custom,
                "parallel" => Tok::Parallel,
                "bus" => Tok::Bus,
                _ => Tok::Ident(word),
            });
        }

        if c.is_ascii_digit() {
            let start = self.pos;
            if c == b'0' && (self.peek_at(1) == b'x' || self.peek_at(1) == b'X') {
                self.bump();
                self.bump();
                let hex_start = self.pos;
                while self.peek().is_ascii_hexdigit() {
                    self.bump();
                }
                if self.pos == hex_start {
                    return Err(format!("{}:{}: `0x` with no hex digits", line, col));
                }
                let text = std::str::from_utf8(&self.src[hex_start..self.pos]).unwrap();
                return BigUint::from_hex_str(text)
                    .map(Tok::Number)
                    .map_err(|e| format!("{}:{}: {}", line, col, e));
            }
            while self.peek().is_ascii_digit() {
                self.bump();
            }
            // A `.` after digits is left to the parser. It is a version in
            // `pragma circom 2.0.0` and a float anywhere else, and only the
            // parser knows which - rejecting it here made every real circom
            // file fail on line 1.
            let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
            return Ok(Tok::Number(BigUint::from_str(text)));
        }

        if c == b'"' {
            self.bump();
            let start = self.pos;
            while self.peek() != b'"' {
                if self.pos >= self.src.len() {
                    return Err(format!("{}:{}: unterminated string", line, col));
                }
                self.bump();
            }
            let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap().to_string();
            self.bump();
            return Ok(Tok::Str(text));
        }

        // Longest-match operators. `<==` must be tried before `<=` before `<`,
        // and `==>` before `==`; getting that order wrong turns a constraint
        // into a comparison, which compiles.
        let three: &[(&[u8; 3], Tok)] = &[
            (b"<==", Tok::AssignConstrainL),
            (b"==>", Tok::AssignConstrainR),
            (b"<--", Tok::AssignL),
            (b"-->", Tok::AssignR),
            (b"===", Tok::ConstrainEq),
            (b"**=", Tok::PowAssign),
            (b"<<=", Tok::ShlAssign),
            (b">>=", Tok::ShrAssign),
        ];
        for (pat, tok) in three {
            if self.src[self.pos..].starts_with(*pat) {
                self.bump();
                self.bump();
                self.bump();
                return Ok(tok.clone());
            }
        }

        let two: &[(&[u8; 2], Tok)] = &[
            (b"==", Tok::Eq),
            (b"!=", Tok::Neq),
            (b"<=", Tok::Le),
            (b">=", Tok::Ge),
            (b"&&", Tok::AndAnd),
            (b"||", Tok::OrOr),
            (b"**", Tok::Pow),
            (b"<<", Tok::Shl),
            (b">>", Tok::Shr),
            (b"+=", Tok::PlusAssign),
            (b"-=", Tok::MinusAssign),
            (b"*=", Tok::StarAssign),
            (b"/=", Tok::SlashAssign),
            (b"\\=", Tok::IntDivAssign),
            (b"%=", Tok::PercentAssign),
            (b"&=", Tok::AndAssign),
            (b"|=", Tok::OrAssign),
            (b"^=", Tok::XorAssign),
            (b"++", Tok::Inc),
            (b"--", Tok::Dec),
        ];
        for (pat, tok) in two {
            if self.src[self.pos..].starts_with(*pat) {
                self.bump();
                self.bump();
                return Ok(tok.clone());
            }
        }

        self.bump();
        Ok(match c {
            b'+' => Tok::Plus,
            b'-' => Tok::Minus,
            b'*' => Tok::Star,
            b'/' => Tok::Slash,
            b'\\' => Tok::IntDiv,
            b'%' => Tok::Percent,
            b'<' => Tok::Lt,
            b'>' => Tok::Gt,
            b'!' => Tok::Not,
            b'&' => Tok::Amp,
            b'|' => Tok::Pipe,
            b'^' => Tok::Caret,
            b'~' => Tok::Tilde,
            b'=' => Tok::Assign,
            b'(' => Tok::LParen,
            b')' => Tok::RParen,
            b'[' => Tok::LBracket,
            b']' => Tok::RBracket,
            b'{' => Tok::LBrace,
            b'}' => Tok::RBrace,
            b',' => Tok::Comma,
            b';' => Tok::Semi,
            b'.' => Tok::Dot,
            b'?' => Tok::Question,
            b':' => Tok::Colon,
            other => {
                return Err(format!(
                    "{}:{}: unexpected character {:?}",
                    line, col, other as char
                ))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(src: &str) -> Vec<Tok> {
        Lexer::new(src).tokenize().unwrap().into_iter().map(|t| t.tok).collect()
    }

    /// The three-character signal operators must win against their two- and
    /// one-character prefixes. `out <== in` lexed as `out < (== in)` parses as
    /// a comparison and compiles to a completely different circuit.
    #[test]
    fn signal_operators_beat_their_prefixes() {
        assert_eq!(toks("a <== b")[1], Tok::AssignConstrainL);
        assert_eq!(toks("a ==> b")[1], Tok::AssignConstrainR);
        assert_eq!(toks("a <-- b")[1], Tok::AssignL);
        assert_eq!(toks("a --> b")[1], Tok::AssignR);
        assert_eq!(toks("a === b")[1], Tok::ConstrainEq);
        assert_eq!(toks("a <= b")[1], Tok::Le);
        assert_eq!(toks("a == b")[1], Tok::Eq);
        assert_eq!(toks("a < b")[1], Tok::Lt);
    }

    /// `/` is FIELD division in circom and `\` is integer division. Confusing
    /// them silently changes the arithmetic.
    #[test]
    fn field_and_integer_division_are_distinct() {
        assert_eq!(toks("a / b")[1], Tok::Slash);
        assert_eq!(toks("a \\ b")[1], Tok::IntDiv);
    }

    #[test]
    fn numbers_decimal_and_hex() {
        assert_eq!(toks("42")[0], Tok::Number(BigUint::from_u64(42)));
        assert_eq!(toks("0x2A")[0], Tok::Number(BigUint::from_u64(42)));
        let big = "21888242871839275222246405745257275088548364400416034343698204186575808495617";
        assert_eq!(toks(big)[0], Tok::Number(BigUint::from_str(big)));
    }

    #[test]
    fn comments_and_pragma() {
        let t = toks("pragma circom 2.0.0; // trailing\n/* block\n */ template T");
        assert_eq!(t[0], Tok::Pragma);
        assert_eq!(t[1], Tok::Ident("circom".into()));
        // `2.0.0` is a version, not a float: number dot number dot number.
        assert_eq!(t[2], Tok::Number(BigUint::from_u64(2)));
        assert_eq!(t[3], Tok::Dot);
        assert!(matches!(t[t.len() - 3], Tok::Template));
    }

    #[test]
    fn unterminated_block_comment_is_an_error() {
        assert!(Lexer::new("/* nope").tokenize().is_err());
    }
}
