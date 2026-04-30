// ============================================================
//  Y-Lang  —  Lexer (Front-End Tokenizer) mapped to Y-Lang
//  lexer.y
// ============================================================

import lib;

// ── Token kinds ──────────────────────────────────────────────

enum TokenKind {
    // ── Keywords ─────────────────────────────────────────────
    Kernel,
    Let,
    Type,
    For,
    In,
    Step,
    Return,
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
    If,
    Else,
    True,
    False,

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
    MmaMod(String), 

    // ── Fragment roles ───────────────────────────────────────
    RoleA, RoleB, RoleC, RoleD,

    // ── Primitive / scalar dtypes ────────────────────────────
    F16, BF16, TF32, F32, F64,
    I8, I16, I32, I64,
    U3, U8, U16, U32, U64,
    Bool, StringTy, CharTy, VecTy, File, OptionTy,

    // ── Transfer policies ────────────────────────────────────
    Async, Sync,

    // ── Cache policies ───────────────────────────────────────
    L2Persist, L2EvictFirst, L2EvictLast, L2Stream,

    // ── Hardware targets ─────────────────────────────────────
    HardwareTarget(String),

    // ── Built-in functions ───────────────────────────────────
    CpAsync, LdMatrix, MmaSync, BarrierSync,

    // ── Attributes (start with @) ────────────────────────────
    AtRequire, AtCachePolicy, AtPtxEmit, AtAvxEmit, AtInline,
    AtNoInline, AtAlign, AtSafe, AtUnsafe, AtGpuUncached,
    AtAtomic, AtStaticAssert, AtUnknown(String),

    // ── Operators ────────────────────────────────────────────
    Arrow, DotDot, ColonColon, Colon, Assign, FatArrow,
    EqEq, NotEq, Lt, Gt, LtEq, GtEq, Plus, Minus, Star,
    Slash, Percent, Ampersand, Pipe, Caret, Bang, AmpAmp,
    PipePipe, LtLt, GtGt, PlusAssign, MinusAssign,
    StarAssign, SlashAssign, Dot,

    // ── Delimiters ───────────────────────────────────────────
    LBrace, RBrace, LParen, RParen, LBracket, RBracket,
    Semicolon, Comma,

    // ── Literals ─────────────────────────────────────────────
    IntLit(I64), FloatLit(F64), StringLit(String),
    CharLit(char),

    // ── Identifiers ──────────────────────────────────────────
    Ident(String),

    // ── Special ──────────────────────────────────────────────
    Eof, Unknown(char)
}

struct Token {
    kind: TokenKind,
    line: usize,
    col: usize,
    lexeme: String
}

impl Token {
    @unsafe
    fn new(kind: TokenKind, line: usize, col: usize, lexeme: &String) -> Token {
        let tok: Token;
        tok.kind = kind;
        tok.line = line;
        tok.col = col;
        tok.lexeme = String::clone(lexeme);
        return tok;
    }
}

struct Lexer {
    input: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    @unsafe
    fn new(source: &String) -> Lexer {
        let lx: Lexer;
        let mut input_vec: Vec<char> = Vec::new(1); // char element size is 1 byte in C, wait no sizeof(char)=1, wait we use `char` primitive which C emitter translates to char
        
        let len: usize = String::len(source);
        let mut i: usize = 0;
        
        while i < len {
            let ch: char = String::char_at(source, i);
            Vec::push(&mut input_vec, ch); // NOTE replaced &ch with ch since it takes char
            i += 1;
        }

        lx.input = input_vec;
        lx.pos = 0;
        lx.line = 1;
        lx.col = 1;
        return lx;
    }

    // Since Y-Lang doesn't have Option types perfectly mapped natively yet,
    // we return a null char '\0' to represent EOF for peek().
    @unsafe
    fn peek(lx: &Lexer) -> char {
        let max_len: usize = Vec::len(&(*lx).input);
        if (*lx).pos >= max_len {
            return '\0';
        }
        return Vec::get_char(&(*lx).input, (*lx).pos);
    }
    
    @unsafe
    fn peek2(lx: &Lexer) -> char {
        let max_len: usize = Vec::len(&(*lx).input);
        if (*lx).pos + 1 >= max_len {
            return '\0';
        }
        return Vec::get_char(&(*lx).input, (*lx).pos + 1);
    }

    @unsafe
    fn advance(lx: &mut Lexer) -> char {
        let ch: char = Lexer::peek(lx);
        if ch == '\0' {
            return ch;
        }
        
        (*lx).pos += 1;
        if ch == '\n' {
            (*lx).line += 1;
            (*lx).col = 1;
        } else {
            (*lx).col += 1;
        }
        
        return ch;
    }

    @unsafe
    fn matches_next(lx: &mut Lexer, expected: char) -> bool {
        let ch: char = Lexer::peek(lx);
        if ch == expected {
            Lexer::advance(lx);
            return true;
        }
        return false;
    }
}

impl Lexer {
    @unsafe
    fn classify_ident(s: &String) -> TokenKind {
        if String::eq_cstr(s, "enum") { return TokenKind::Enum; }
        if String::eq_cstr(s, "struct") { return TokenKind::Struct; }
        if String::eq_cstr(s, "impl") { return TokenKind::Impl; }
        if String::eq_cstr(s, "fn") { return TokenKind::Fn; }
        if String::eq_cstr(s, "let") { return TokenKind::Let; }
        if String::eq_cstr(s, "return") { return TokenKind::Return; }
        if String::eq_cstr(s, "if") { return TokenKind::If; }
        if String::eq_cstr(s, "else") { return TokenKind::Else; }
        if String::eq_cstr(s, "while") { return TokenKind::While; }
        if String::eq_cstr(s, "true") { return TokenKind::True; }
        if String::eq_cstr(s, "false") { return TokenKind::False; }
        if String::eq_cstr(s, "mut") { return TokenKind::Mut; }
        
        // Types
        if String::eq_cstr(s, "String") { return TokenKind::StringTy; }
        if String::eq_cstr(s, "Vec") { return TokenKind::VecTy; }
        if String::eq_cstr(s, "char") { return TokenKind::CharTy; }
        if String::eq_cstr(s, "usize") { return TokenKind::Ident(String::clone(s)); }
        if String::eq_cstr(s, "I32") { return TokenKind::I32; }
        if String::eq_cstr(s, "I64") { return TokenKind::I64; }
        if String::eq_cstr(s, "bool") { return TokenKind::Bool; }
        
        // Ident
        return TokenKind::Ident(String::clone(s));
    }
    
    @unsafe
    fn skip_whitespace(lx: &mut Lexer) {
        let mut parsing: bool = true;
        while parsing {
            let ch: char = Lexer::peek(lx);
            if ch == ' ' { Lexer::advance(lx); }
            else if ch == '\n' { Lexer::advance(lx); }
            else if ch == '\r' { Lexer::advance(lx); }
            else if ch == '\t' { Lexer::advance(lx); }
            else { parsing = false; }
        }
    }

    @unsafe
    fn scan_ident_or_keyword(lx: &mut Lexer, start_col: usize, first_char: char) -> Token {
        let line: usize = (*lx).line;
        let mut s: String = String::new("");
        String::push(&mut s, first_char);
        
        let mut parsing: bool = true;
        while parsing {
            let ch: char = Lexer::peek(lx);
            let mut is_alpha: bool = false;
            if ch >= 'a' { if ch <= 'z' { is_alpha = true; } }
            if ch >= 'A' { if ch <= 'Z' { is_alpha = true; } }
            if ch >= '0' { if ch <= '9' { is_alpha = true; } }
            if ch == '_' { is_alpha = true; }
            
            if is_alpha {
                String::push(&mut s, ch);
                Lexer::advance(lx);
            } else {
                parsing = false;
            }
        }
        
        let kind: TokenKind = Lexer::classify_ident(&s);
        return Token::new(kind, line, start_col, &s);
    }
}

impl Lexer {
    @unsafe
    fn scan_number(lx: &mut Lexer, start_col: usize, first_char: char) -> Token {
        let line: usize = (*lx).line;
        let mut s: String = String::new("");
        String::push(&mut s, first_char);
        
        let mut is_float: bool = false;
        let mut parsing: bool = true;
        
        while parsing {
            let ch: char = Lexer::peek(lx);
            let mut is_digit: bool = false;
            if ch >= '0' { if ch <= '9' { is_digit = true; } }
            
            if is_digit {
                String::push(&mut s, ch);
                Lexer::advance(lx);
            } else if ch == '.' {
                is_float = true;
                String::push(&mut s, ch);
                Lexer::advance(lx);
            } else {
                parsing = false;
            }
        }
        
        if is_float {
            return Token::new(TokenKind::FloatLit(0.0), line, start_col, &s);
        } else {
            return Token::new(TokenKind::IntLit(0), line, start_col, &s);
        }
    }

    @unsafe
    fn next_token(lx: &mut Lexer) -> Token {
        Lexer::skip_whitespace(lx);

        let line: usize = (*lx).line;
        let start_col: usize = (*lx).col;

        let ch: char = Lexer::advance(lx);
        
        if ch == '\0' {
            let empty: String = String::new("");
            return Token::new(TokenKind::Eof, line, start_col, &empty);
        }

        // Single-char operators
        if ch == '{' { let x: String = String::new("{"); return Token::new(TokenKind::LBrace, line, start_col, &x); }
        if ch == '}' { let x: String = String::new("}"); return Token::new(TokenKind::RBrace, line, start_col, &x); }
        if ch == '(' { let x: String = String::new("("); return Token::new(TokenKind::LParen, line, start_col, &x); }
        if ch == ')' { let x: String = String::new(")"); return Token::new(TokenKind::RParen, line, start_col, &x); }
        if ch == ';' { let x: String = String::new(";"); return Token::new(TokenKind::Semicolon, line, start_col, &x); }
        if ch == ':' { let x: String = String::new(":"); return Token::new(TokenKind::Colon, line, start_col, &x); }
        if ch == ',' { let x: String = String::new(","); return Token::new(TokenKind::Comma, line, start_col, &x); }
        if ch == '=' { let x: String = String::new("="); return Token::new(TokenKind::Assign, line, start_col, &x); }
        if ch == '+' { let x: String = String::new("+"); return Token::new(TokenKind::Plus, line, start_col, &x); }
        if ch == '*' { let x: String = String::new("*"); return Token::new(TokenKind::Star, line, start_col, &x); }
        if ch == '&' { let x: String = String::new("&"); return Token::new(TokenKind::Ampersand, line, start_col, &x); }
        if ch == '<' { let x: String = String::new("<"); return Token::new(TokenKind::Lt, line, start_col, &x); }
        if ch == '>' { let x: String = String::new(">"); return Token::new(TokenKind::Gt, line, start_col, &x); }

        let mut is_alpha: bool = false;
        if ch >= 'a' { if ch <= 'z' { is_alpha = true; } }
        if ch >= 'A' { if ch <= 'Z' { is_alpha = true; } }
        if ch == '_' { is_alpha = true; }

        if is_alpha {
            return Lexer::scan_ident_or_keyword(lx, start_col, ch);
        }

        let mut is_digit: bool = false;
        if ch >= '0' { if ch <= '9' { is_digit = true; } }
        
        if is_digit {
            return Lexer::scan_number(lx, start_col, ch);
        }

        let other: String = String::new("?");
        return Token::new(TokenKind::Unknown(ch), line, start_col, &other);
    }
}

impl Lexer {
    @unsafe
    fn tokenize(lx: &mut Lexer) -> Vec {
        let mut tokens: Vec = Vec::new(32);
        let mut parsing: bool = true;
        
        while parsing {
            let tok: Token = Lexer::next_token(lx);
            
            let len: usize = String::len(&tok.lexeme);
            if len == 0 {
                parsing = false;
            } else {
                Vec::push(&mut tokens, tok);
            }
        }
        
        return tokens;
    }
}

@unsafe
fn main() {
    println("--- Y-Lang Self-Hosted Lexer Bootstrapping ---");
    let source: String = String::new("fn test() ; let x = 123 }");
    
    let mut lexer: Lexer = Lexer::new(&source);
    
    let mut parsing: bool = true;
    while parsing {
        let tok: Token = Lexer::next_token(&mut lexer);
        
        let len: usize = String::len(&tok.lexeme);
        if len == 0 {
            println("[EOF]");
            parsing = false;
        } else {
            print("Token: ");
            println(&tok.lexeme); // Fixed print arg to reference
        }
    }
    
    return;
}
