// ============================================================
//  Y  —  Lexer (Front-End Tokenizer)
//  Subagent C | lexer.rs
//
//  Tokenizes Y source code into a flat stream of Tokens.
//  Covers every construct defined in Y_Specification v0.1:
//    - Keywords, types, dtypes, attributes
//    - Operators (including -> and ::)
//    - Identifiers, integer literals, float literals, strings
//    - Generic angle brackets < >
//    - Line and block comments (stripped)
//    - Hardware targets, cache policies, MMA ops
// ============================================================

#![allow(dead_code)]

// ────────────────────────────────────────────────────────────
//  Token kinds
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // ── Keywords ─────────────────────────────────────────────
    Kernel,
    Let,
    Type,
    For,
    In,
    Step,
    Return,
    If,
    Else,
    True,
    False,
    Const,
    Pub,
    Mut,
    Unsafe,
    Wait,
    Emit,
    Load,
    Store,
    Barrier,
    Struct,
    Enum,
    Fn,
    Match,
    Safe,
    Chisel,
    Import,
    While,
    Impl,
    SelfKw,

    // ── Memory space types ───────────────────────────────────
    GlobalMemory,
    L2Memory,
    SharedMemory,
    RegisterFile,

    // ── Layout / transfer / fragment types ──────────────────
    SmemLayout,
    Swizzle,
    NoSwizzle,
    Pipeline,
    Transfer,
    Fragment,

    // ── MMA atom identifiers ─────────────────────────────────
    MmaMod(String), // MMA_m16n8k16, etc.

    // ── Fragment roles ───────────────────────────────────────
    RoleA,
    RoleB,
    RoleC,
    RoleD,

    // ── Primitive / scalar dtypes ────────────────────────────
    F16,
    BF16,
    TF32,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U3,
    U8,
    U16,
    U32,
    U64,
    QFixed(String), // e.g. "Q32.32"
    Bool,
    StringTy,
    CharTy,
    VecTy,
    File,
    OptionTy,

    // ── Transfer policies ────────────────────────────────────
    Async,
    Sync,

    // ── Cache policies ───────────────────────────────────────
    L2Persist,
    L2EvictFirst,
    L2EvictLast,
    L2Stream,

    // ── Hardware targets ─────────────────────────────────────
    HardwareTarget(String), // RTX_4070, H100, CPU_AVX512, …

    // ── Built-in functions ───────────────────────────────────
    CpAsync,
    LdMatrix,
    MmaSync,
    BarrierSync,

    // ── Attributes (start with @) ────────────────────────────
    AtRequire,
    AtCachePolicy,
    AtPtxEmit,
    AtAvxEmit,
    AtInline,
    AtNoInline,
    AtAlign,
    AtSafe,
    AtUnsafe,
    AtGpuUncached,
    AtAtomic,
    AtStaticAssert,
    AtZeroDrift,       // @ZeroDrift — enforce zero numerical drift
    AtBounds,
    AtInvariant,
    AtDivergenceUniform,
    AtUnknown(String), // future-proof

    // ── Operators ────────────────────────────────────────────
    Arrow,       // ->
    DotDot,      // ..
    ColonColon,  // ::
    Colon,       // :
    Assign,      // =
    FatArrow,    // =>
    EqEq,        // ==
    NotEq,       // !=
    Lt,          // <
    Gt,          // >
    LtEq,        // <=
    GtEq,        // >=
    Plus,        // +
    Minus,       // -
    Star,        // *
    Slash,       // /
    Percent,     // %
    Ampersand,   // &
    Pipe,        // |
    Caret,       // ^
    Bang,        // !
    AmpAmp,      // &&
    PipePipe,    // ||
    LtLt,        // <<
    GtGt,        // >>
    PlusAssign,  // +=
    MinusAssign, // -=
    StarAssign,  // *=
    SlashAssign, // /=
    Dot,         // .

    // ── Delimiters ───────────────────────────────────────────
    LBrace,    // {
    RBrace,    // }
    LParen,    // (
    RParen,    // )
    LBracket,  // [
    RBracket,  // ]
    Semicolon, // ;
    Comma,     // ,

    // ── Literals ─────────────────────────────────────────────
    /// An integer literal
    IntLit(i64),
    /// A floating-point literal
    FloatLit(f64),
    /// A quoted string literal
    StringLit(String),
    /// A character literal like 'c'
    CharLit(char),

    // ── Identifiers ──────────────────────────────────────────
    Ident(String),

    // ── Special ──────────────────────────────────────────────
    Eof,
    Unknown(char),
}

// ────────────────────────────────────────────────────────────
//  Token  —  a kind + source location
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub line: usize,
    pub col: usize,
    /// The raw text slice that produced this token.
    pub lexeme: String,
}

impl Token {
    pub fn new(kind: TokenKind, line: usize, col: usize, lexeme: impl Into<String>) -> Self {
        Self {
            kind,
            line,
            col,
            lexeme: lexeme.into(),
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Lexer
// ────────────────────────────────────────────────────────────

pub struct Lexer {
    input: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Self {
            input: source.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    // ── Internal helpers ─────────────────────────────────────

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<char> {
        self.input.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.input.get(self.pos).copied()?;
        self.pos += 1;
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }

    fn matches_next(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_line_comment(&mut self) {
        // consume until newline or EOF
        while let Some(ch) = self.peek() {
            self.advance();
            if ch == '\n' {
                break;
            }
        }
    }

    fn skip_block_comment(&mut self) {
        // already consumed '/*'; skip until '*/'
        loop {
            match self.advance() {
                None => break, // unterminated — let parser error
                Some('*') if self.peek() == Some('/') => {
                    self.advance();
                    break;
                }
                _ => {}
            }
        }
    }

    // ── Numeric literal scanning ──────────────────────────────

    fn scan_number(&mut self, start_col: usize, first: char) -> Token {
        let line = self.line;
        let mut lexeme = String::from(first);
        let mut is_float = false;

        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                lexeme.push(ch);
                self.advance();
            } else if ch == '.' && self.peek2().map_or(false, |c| c.is_ascii_digit()) {
                is_float = true;
                lexeme.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if is_float {
            let val: f64 = lexeme.parse().unwrap_or(0.0);
            Token::new(TokenKind::FloatLit(val), line, start_col, &lexeme)
        } else {
            let val: i64 = lexeme.parse().unwrap_or(0);
            Token::new(TokenKind::IntLit(val), line, start_col, &lexeme)
        }
    }

    // ── String literal scanning ───────────────────────────────

    fn scan_string(&mut self, start_col: usize) -> Token {
        let line = self.line;
        let mut s = String::new();
        loop {
            match self.advance() {
                None | Some('"') => break,
                Some('\\') => {
                    if let Some(esc) = self.advance() {
                        match esc {
                            'n' => s.push('\n'),
                            't' => s.push('\t'),
                            '"' => s.push('"'),
                            '\\' => s.push('\\'),
                            other => {
                                s.push('\\');
                                s.push(other);
                            }
                        }
                    }
                }
                Some(ch) => s.push(ch),
            }
        }
        let lexeme = format!("\"{}\"", s);
        Token::new(TokenKind::StringLit(s), line, start_col, lexeme)
    }

    // ── Identifier / keyword scanning ────────────────────────

    fn scan_ident_or_keyword(&mut self, start_col: usize, first: char) -> Token {
        let line = self.line;
        let mut ident = String::from(first);

        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                self.advance();
            } else if ch == '.' && first == 'Q' && ident.len() > 1 {
                // Fixed-point type support: e.g., "Q32" + "." + "32"
                if self.peek2().map_or(false, |c| c.is_ascii_digit()) {
                    ident.push(ch);
                    self.advance();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        let kind = Self::classify_ident(&ident);
        Token::new(kind, line, start_col, &ident)
    }

    /// Map a raw identifier string to its TokenKind.
    fn classify_ident(s: &str) -> TokenKind {
        match s {
            // Keywords
            "kernel" => TokenKind::Kernel,
            "let" => TokenKind::Let,
            "type" => TokenKind::Type,
            "for" => TokenKind::For,
            "in" => TokenKind::In,
            "step" => TokenKind::Step,
            "return" => TokenKind::Return,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "const" => TokenKind::Const,
            "pub" => TokenKind::Pub,
            "mut" => TokenKind::Mut,
            "unsafe" => TokenKind::Unsafe,
            "wait" => TokenKind::Wait,
            "emit" => TokenKind::Emit,
            "load" => TokenKind::Load,
            "store" => TokenKind::Store,
            "barrier" => TokenKind::Barrier,
            "struct" => TokenKind::Struct,
            "enum" => TokenKind::Enum,
            "fn" => TokenKind::Fn,
            "match" => TokenKind::Match,
            "safe" => TokenKind::Safe,
            "chisel" => TokenKind::Chisel,
            "import" => TokenKind::Import,
            "func" => TokenKind::Fn,
            "while" => TokenKind::While,
            "impl" => TokenKind::Impl,
            "self" => TokenKind::SelfKw,

            // Memory space types
            "GlobalMemory" => TokenKind::GlobalMemory,
            "L2Memory" => TokenKind::L2Memory,
            "SharedMemory" => TokenKind::SharedMemory,
            "RegisterFile" => TokenKind::RegisterFile,

            // Layout / structural types
            "SmemLayout" => TokenKind::SmemLayout,
            "Swizzle" => TokenKind::Swizzle,
            "NoSwizzle" => TokenKind::NoSwizzle,
            "Pipeline" => TokenKind::Pipeline,
            "Transfer" => TokenKind::Transfer,
            "Fragment" => TokenKind::Fragment,

            // Fragment roles (single-char uppercase — checked AFTER MMA_*)
            "A" => TokenKind::RoleA,
            "B" => TokenKind::RoleB,
            "C" => TokenKind::RoleC,
            "D" => TokenKind::RoleD,

            // Dtypes
            "F16" => TokenKind::F16,
            "BF16" => TokenKind::BF16,
            "TF32" => TokenKind::TF32,
            "F32" => TokenKind::F32,
            "F64" => TokenKind::F64,
            "I8" => TokenKind::I8,
            "I16" => TokenKind::I16,
            "I32" => TokenKind::I32,
            "I64" => TokenKind::I64,
            "U3" => TokenKind::U3,
            "U8" => TokenKind::U8,
            "U16" => TokenKind::U16,
            "U32" => TokenKind::U32,
            "U64" => TokenKind::U64,
            "u3" => TokenKind::U3,
            "u8" => TokenKind::U8,
            "u16" => TokenKind::U16,
            "u32" => TokenKind::U32,
            "u64" => TokenKind::U64,
            "i8" => TokenKind::I8,
            "i16" => TokenKind::I16,
            "i32" => TokenKind::I32,
            "i64" => TokenKind::I64,
            "f16" => TokenKind::F16,
            "f32" => TokenKind::F32,
            "f64" => TokenKind::F64,
            "bool" => TokenKind::Bool,
            "String" => TokenKind::StringTy,
            "char" => TokenKind::CharTy,
            "Vec" => TokenKind::VecTy,
            "File" => TokenKind::File,
            "Option" => TokenKind::OptionTy,

            // Transfer policies
            "Async" => TokenKind::Async,
            "Sync" => TokenKind::Sync,

            // Cache policies
            "L2_PERSIST" => TokenKind::L2Persist,
            "L2_EVICT_FIRST" => TokenKind::L2EvictFirst,
            "L2_EVICT_LAST" => TokenKind::L2EvictLast,
            "L2_STREAM" => TokenKind::L2Stream,

            // Hardware targets
            "RTX_4070" | "RTX_4090" | "RTX_3090" | "H100" | "A100" | "CPU_AVX2" | "CPU_AVX512" => {
                TokenKind::HardwareTarget(s.to_string())
            }

            // Built-in functions
            "cp_async" => TokenKind::CpAsync,
            "ldmatrix" => TokenKind::LdMatrix,
            "mma_sync" => TokenKind::MmaSync,

            // MMA atom names   MMA_m16n8k16 etc.
            s if s.starts_with("MMA_") => TokenKind::MmaMod(s.to_string()),

            // Fixed-point dtypes e.g. Q32.32
            s if s.starts_with('Q') && s.contains('.') => TokenKind::QFixed(s.to_string()),

            // Everything else is an identifier
            _ => TokenKind::Ident(s.to_string()),
        }
    }

    // ── Attribute scanning (@target, @cache_policy, …) ───────

    fn scan_attribute(&mut self, start_col: usize) -> Token {
        let line = self.line;
        let mut name = String::from('@');
        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                name.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        let mut kind = match name.as_str() {
            "@require" => TokenKind::AtRequire,
            "@cache_policy" => TokenKind::AtCachePolicy,
            "@ptx_emit" => TokenKind::AtPtxEmit,
            "@avx_emit" => TokenKind::AtAvxEmit,
            "@inline" => TokenKind::AtInline,
            "@noinline" => TokenKind::AtNoInline,
            "@align" => TokenKind::AtAlign,
            "@safe" => TokenKind::AtSafe,
            "@unsafe" => TokenKind::AtUnsafe,
            "@gpu_uncached" => TokenKind::AtGpuUncached,
            "@atomic" => TokenKind::AtAtomic,
            "@static_assert" => TokenKind::AtStaticAssert,
            "@ZeroDrift" => TokenKind::AtZeroDrift,
            "@bounds" => TokenKind::AtBounds,
            "@invariant" => TokenKind::AtInvariant,
            "@divergence" => {
                // Peek for (uniform)
                if self.peek() == Some('(') {
                    self.advance();
                    let mut arg = String::new();
                    while let Some(ch) = self.peek() {
                        if ch.is_alphabetic() {
                            arg.push(ch);
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    if self.peek() == Some(')') {
                        self.advance();
                        if arg == "uniform" {
                            name.push_str("(uniform)");
                            TokenKind::AtDivergenceUniform
                        } else {
                            name.push('(');
                            name.push_str(&arg);
                            name.push(')');
                            TokenKind::AtUnknown(name.clone())
                        }
                    } else {
                        TokenKind::AtUnknown(name.clone())
                    }
                } else {
                    TokenKind::AtUnknown(name.clone())
                }
            }
            other => TokenKind::AtUnknown(other.to_string()),
        };
        Token::new(kind, line, start_col, &name)
    }

    // ── Main next-token function ──────────────────────────────

    fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        let line = self.line;
        let start_col = self.col;

        let ch = match self.advance() {
            None => return Token::new(TokenKind::Eof, line, start_col, ""),
            Some(ch) => ch,
        };

        match ch {
            // ── Comments ──────────────────────────────────────
            '/' if self.peek() == Some('/') => {
                self.skip_line_comment();
                self.next_token() // recurse past comment
            }
            '/' if self.peek() == Some('*') => {
                self.advance();
                self.skip_block_comment();
                self.next_token()
            }

            // ── Attribute ─────────────────────────────────────
            '@' => self.scan_attribute(start_col),

            // ── String literal ────────────────────────────────
            '"' => self.scan_string(start_col),

            // ── Character literal ─────────────────────────────
            '\'' => {
                let tok_char = match self.advance() {
                    Some('\\') => match self.advance() {
                        Some('n') => '\n',
                        Some('r') => '\r',
                        Some('t') => '\t',
                        Some('0') => '\0',
                        Some('\\') => '\\',
                        Some('\'') => '\'',
                        Some(c) => c,
                        None => {
                            return Token::new(TokenKind::Unknown('\\'), line, start_col, "'\\")
                        }
                    },
                    Some(c) => c,
                    None => return Token::new(TokenKind::Unknown('\''), line, start_col, "'"),
                };
                if self.peek() == Some('\'') {
                    self.advance();
                }
                let lexeme = if tok_char == '\n' {
                    "'\\n'".to_string()
                } else if tok_char == '\r' {
                    "'\\r'".to_string()
                } else if tok_char == '\t' {
                    "'\\t'".to_string()
                } else if tok_char == '\0' {
                    "'\\0'".to_string()
                } else if tok_char == '\\' {
                    "'\\\\'".to_string()
                } else if tok_char == '\'' {
                    "'\\''".to_string()
                } else {
                    format!("'{}'", tok_char)
                };
                Token::new(TokenKind::CharLit(tok_char), line, start_col, &lexeme)
            }

            // ── Numeric literal ───────────────────────────────
            c if c.is_ascii_digit() => self.scan_number(start_col, c),

            // ── Identifier / keyword ─────────────────────────
            c if c.is_alphabetic() || c == '_' => self.scan_ident_or_keyword(start_col, c),

            // ── Two-char operators first ──────────────────────
            '-' if self.peek() == Some('>') => {
                self.advance();
                Token::new(TokenKind::Arrow, line, start_col, "->")
            }
            '.' if self.peek() == Some('.') => {
                self.advance();
                Token::new(TokenKind::DotDot, line, start_col, "..")
            }
            ':' if self.peek() == Some(':') => {
                self.advance();
                Token::new(TokenKind::ColonColon, line, start_col, "::")
            }
            '=' if self.peek() == Some('>') => {
                self.advance();
                Token::new(TokenKind::FatArrow, line, start_col, "=>")
            }
            '=' if self.peek() == Some('=') => {
                self.advance();
                Token::new(TokenKind::EqEq, line, start_col, "==")
            }
            '!' if self.peek() == Some('=') => {
                self.advance();
                Token::new(TokenKind::NotEq, line, start_col, "!=")
            }
            '<' if self.peek() == Some('=') => {
                self.advance();
                Token::new(TokenKind::LtEq, line, start_col, "<=")
            }
            '<' if self.peek() == Some('<') => {
                self.advance();
                Token::new(TokenKind::LtLt, line, start_col, "<<")
            }
            '>' if self.peek() == Some('=') => {
                self.advance();
                Token::new(TokenKind::GtEq, line, start_col, ">=")
            }
            '>' if self.peek() == Some('>') => {
                self.advance();
                Token::new(TokenKind::GtGt, line, start_col, ">>")
            }
            '&' if self.peek() == Some('&') => {
                self.advance();
                Token::new(TokenKind::AmpAmp, line, start_col, "&&")
            }
            '|' if self.peek() == Some('|') => {
                self.advance();
                Token::new(TokenKind::PipePipe, line, start_col, "||")
            }
            '+' if self.peek() == Some('=') => {
                self.advance();
                Token::new(TokenKind::PlusAssign, line, start_col, "+=")
            }
            '-' if self.peek() == Some('=') => {
                self.advance();
                Token::new(TokenKind::MinusAssign, line, start_col, "-=")
            }
            '*' if self.peek() == Some('=') => {
                self.advance();
                Token::new(TokenKind::StarAssign, line, start_col, "*=")
            }
            '/' if self.peek() == Some('=') => {
                self.advance();
                Token::new(TokenKind::SlashAssign, line, start_col, "/=")
            }

            // ── Single-char operators / delimiters ────────────
            ':' => Token::new(TokenKind::Colon, line, start_col, ":"),
            '=' => Token::new(TokenKind::Assign, line, start_col, "="),
            '<' => Token::new(TokenKind::Lt, line, start_col, "<"),
            '>' => Token::new(TokenKind::Gt, line, start_col, ">"),
            '+' => Token::new(TokenKind::Plus, line, start_col, "+"),
            '-' => Token::new(TokenKind::Minus, line, start_col, "-"),
            '*' => Token::new(TokenKind::Star, line, start_col, "*"),
            '/' => Token::new(TokenKind::Slash, line, start_col, "/"),
            '%' => Token::new(TokenKind::Percent, line, start_col, "%"),
            '&' => Token::new(TokenKind::Ampersand, line, start_col, "&"),
            '|' => Token::new(TokenKind::Pipe, line, start_col, "|"),
            '^' => Token::new(TokenKind::Caret, line, start_col, "^"),
            '!' => Token::new(TokenKind::Bang, line, start_col, "!"),
            '.' => Token::new(TokenKind::Dot, line, start_col, "."),
            '{' => Token::new(TokenKind::LBrace, line, start_col, "{"),
            '}' => Token::new(TokenKind::RBrace, line, start_col, "}"),
            '(' => Token::new(TokenKind::LParen, line, start_col, "("),
            ')' => Token::new(TokenKind::RParen, line, start_col, ")"),
            '[' => Token::new(TokenKind::LBracket, line, start_col, "["),
            ']' => Token::new(TokenKind::RBracket, line, start_col, "]"),
            ';' => Token::new(TokenKind::Semicolon, line, start_col, ";"),
            ',' => Token::new(TokenKind::Comma, line, start_col, ","),

            other => Token::new(
                TokenKind::Unknown(other),
                line,
                start_col,
                other.to_string(),
            ),
        }
    }

    // ── Public API ────────────────────────────────────────────

    /// Tokenize the entire source into a Vec<Token>.
    /// The last token is always Eof.
    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token();
            let done = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if done {
                break;
            }
        }
        tokens
    }
}

// ────────────────────────────────────────────────────────────
//  Pretty-printer (for debug / REPL)
// ────────────────────────────────────────────────────────────

pub fn print_tokens(tokens: &[Token]) {
    for t in tokens {
        println!("{:4}:{:3}  {:30}  {:?}", t.line, t.col, t.lexeme, t.kind);
    }
}

// ────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> Vec<TokenKind> {
        Lexer::new(src)
            .tokenize()
            .into_iter()
            .map(|t| t.kind)
            .collect()
    }

    #[test]
    fn test_tokenize_basic() {
        let mut lexer = Lexer::new("let a = 1;");
        let tokens = lexer.tokenize();
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0].kind, TokenKind::Let);
        assert_eq!(tokens[1].kind, TokenKind::Ident("a".to_string()));
        assert_eq!(tokens[2].kind, TokenKind::Assign);
        assert_eq!(tokens[3].kind, TokenKind::IntLit(1));
        assert_eq!(tokens[4].kind, TokenKind::Semicolon);
        assert_eq!(tokens[5].kind, TokenKind::Eof);
    }

    #[test]
    fn test_tokenize() {
        let mut lexer = Lexer::new("let x = 42;");
        let tokens = lexer.tokenize();

        assert_eq!(tokens.len(), 6);

        assert_eq!(tokens[0].kind, TokenKind::Let);
        assert_eq!(tokens[0].lexeme, "let");
        assert_eq!(tokens[0].line, 1);
        assert_eq!(tokens[0].col, 1);

        assert_eq!(tokens[1].kind, TokenKind::Ident("x".to_string()));
        assert_eq!(tokens[1].lexeme, "x");
        assert_eq!(tokens[1].line, 1);
        assert_eq!(tokens[1].col, 5);

        assert_eq!(tokens[2].kind, TokenKind::Assign);
        assert_eq!(tokens[2].lexeme, "=");
        assert_eq!(tokens[2].line, 1);
        assert_eq!(tokens[2].col, 7);

        assert_eq!(tokens[3].kind, TokenKind::IntLit(42));
        assert_eq!(tokens[3].lexeme, "42");
        assert_eq!(tokens[3].line, 1);
        assert_eq!(tokens[3].col, 9);

        assert_eq!(tokens[4].kind, TokenKind::Semicolon);
        assert_eq!(tokens[4].lexeme, ";");
        assert_eq!(tokens[4].line, 1);
        assert_eq!(tokens[4].col, 11);

        assert_eq!(tokens[5].kind, TokenKind::Eof);
        assert_eq!(tokens[5].lexeme, "");
    }

    #[test]
    fn test_tokenize_empty() {
        let mut lexer = Lexer::new("");
        let tokens = lexer.tokenize();

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
    }

    #[test]
    fn test_tokenize_complex() {
        let mut lexer = Lexer::new("let x = 42; // comment\n y -> z;");
        let tokens = lexer.tokenize();

        assert_eq!(tokens.len(), 10);

        assert_eq!(tokens[0].kind, TokenKind::Let);
        assert_eq!(tokens[1].kind, TokenKind::Ident("x".to_string()));
        assert_eq!(tokens[2].kind, TokenKind::Assign);
        assert_eq!(tokens[3].kind, TokenKind::IntLit(42));
        assert_eq!(tokens[4].kind, TokenKind::Semicolon);
        assert_eq!(tokens[5].kind, TokenKind::Ident("y".to_string()));
        assert_eq!(tokens[6].kind, TokenKind::Arrow);
        assert_eq!(tokens[7].kind, TokenKind::Ident("z".to_string()));
        assert_eq!(tokens[8].kind, TokenKind::Semicolon);
        assert_eq!(tokens[9].kind, TokenKind::Eof);
    }

    #[test]
    fn test_kernel_keyword() {
        let kinds = lex("kernel matmul");
        assert_eq!(kinds[0], TokenKind::Kernel);
        assert_eq!(kinds[1], TokenKind::Ident("matmul".to_string()));
    }

    #[test]
    fn test_attribute() {
        let kinds = lex("@require(sm >= 89)");
        assert_eq!(kinds[0], TokenKind::AtRequire);
        assert_eq!(kinds[1], TokenKind::LParen);
        assert_eq!(kinds[2], TokenKind::Ident("sm".to_string()));
    }

    #[test]
    fn test_memory_types() {
        let kinds = lex("GlobalMemory L2Memory SharedMemory RegisterFile");
        assert_eq!(kinds[0], TokenKind::GlobalMemory);
        assert_eq!(kinds[1], TokenKind::L2Memory);
        assert_eq!(kinds[2], TokenKind::SharedMemory);
        assert_eq!(kinds[3], TokenKind::RegisterFile);
    }

    #[test]
    fn test_arrow_and_range() {
        let kinds = lex("0..K -> C");
        assert_eq!(kinds[0], TokenKind::IntLit(0));
        assert_eq!(kinds[1], TokenKind::DotDot);
        assert_eq!(kinds[2], TokenKind::Ident("K".to_string()));
        assert_eq!(kinds[3], TokenKind::Arrow);
    }

    #[test]
    fn test_line_comment_skipped() {
        let kinds = lex("let // this is a comment\n x");
        assert_eq!(kinds[0], TokenKind::Let);
        assert_eq!(kinds[1], TokenKind::Ident("x".to_string()));
    }

    #[test]
    fn test_fragment_type() {
        let kinds = lex("Fragment<MMA_m16n8k16, A, F16>");
        assert_eq!(kinds[0], TokenKind::Fragment);
        assert_eq!(kinds[2], TokenKind::MmaMod("MMA_m16n8k16".into()));
        assert_eq!(kinds[4], TokenKind::RoleA);
        assert_eq!(kinds[6], TokenKind::F16);
    }

    #[test]
    fn test_cache_policy_attr() {
        let kinds = lex("@cache_policy(L2_PERSIST, reuse_count=8)");
        assert_eq!(kinds[0], TokenKind::AtCachePolicy);
        assert_eq!(kinds[2], TokenKind::L2Persist);
    }

    #[test]
    fn test_pipeline_type() {
        let kinds = lex("Pipeline<stages=2, layout=ATile>");
        assert_eq!(kinds[0], TokenKind::Pipeline);
    }

    #[test]
    fn test_float_literal() {
        let kinds = lex("3.14");
        match &kinds[0] {
            TokenKind::FloatLit(v) => assert!((*v - 3.14).abs() < 1e-9),
            other => panic!("expected FloatLit, got {:?}", other),
        }
    }

    #[test]
    fn test_block_comment_skipped() {
        let kinds = lex("let /* skip this */ x");
        assert_eq!(kinds[0], TokenKind::Let);
        assert_eq!(kinds[1], TokenKind::Ident("x".into()));
    }

    #[test]
    fn test_unterminated_block_comment() {
        let kinds = lex("/* unterminated comment");
        assert_eq!(kinds[0], TokenKind::Eof);
    }

    #[test]
    fn test_eof() {
        let kinds = lex("");
        assert_eq!(kinds[0], TokenKind::Eof);
    }

    #[test]
    fn test_unterminated_string() {
        let kinds = lex("\"unterminated");
        assert_eq!(kinds.len(), 2);
        assert_eq!(kinds[0], TokenKind::StringLit("unterminated".to_string()));
        assert_eq!(kinds[1], TokenKind::Eof);
    }

    #[test]
    fn test_tokenize_lines_and_columns() {
        let mut lexer = Lexer::new("let a = 1;\nlet b = 2;");
        let tokens = lexer.tokenize();

        // "let"
        assert_eq!(tokens[0].kind, TokenKind::Let);
        assert_eq!(tokens[0].line, 1);
        assert_eq!(tokens[0].col, 1);

        // "a"
        assert_eq!(tokens[1].kind, TokenKind::Ident("a".to_string()));
        assert_eq!(tokens[1].line, 1);
        assert_eq!(tokens[1].col, 5);

        // "="
        assert_eq!(tokens[2].kind, TokenKind::Assign);
        assert_eq!(tokens[2].line, 1);
        assert_eq!(tokens[2].col, 7);

        // "1"
        assert_eq!(tokens[3].kind, TokenKind::IntLit(1));
        assert_eq!(tokens[3].line, 1);
        assert_eq!(tokens[3].col, 9);

        // ";"
        assert_eq!(tokens[4].kind, TokenKind::Semicolon);
        assert_eq!(tokens[4].line, 1);
        assert_eq!(tokens[4].col, 10);

        // "let"
        assert_eq!(tokens[5].kind, TokenKind::Let);
        assert_eq!(tokens[5].line, 2);
        assert_eq!(tokens[5].col, 1);

        // "b"
        assert_eq!(tokens[6].kind, TokenKind::Ident("b".to_string()));
        assert_eq!(tokens[6].line, 2);
        assert_eq!(tokens[6].col, 5);
    }

    #[test]
    fn test_tokenize_unknown_characters() {
        let mut lexer = Lexer::new("let $ = 1;");
        let tokens = lexer.tokenize();

        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0].kind, TokenKind::Let);
        assert_eq!(tokens[1].kind, TokenKind::Unknown('$'));
        assert_eq!(tokens[2].kind, TokenKind::Assign);
        assert_eq!(tokens[3].kind, TokenKind::IntLit(1));
        assert_eq!(tokens[4].kind, TokenKind::Semicolon);
        assert_eq!(tokens[5].kind, TokenKind::Eof);
    }
}

