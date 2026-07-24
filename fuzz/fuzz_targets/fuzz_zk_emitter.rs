#![no_main]
use libfuzzer_sys::fuzz_target;
use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;
use y::zk_emitter::ZkEmitter;

fuzz_target!(|data: &[u8]| {
    let source = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return,
    };

    // 1. Lexing
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();
    if tokens.is_empty() {
        return;
    }

    // 2. Parsing
    let mut parser = Parser::new(tokens);
    let program = match parser.parse_program() {
        Ok(p) => p,
        Err(_) => return, // Ignore standard syntax errors
    };

    // 3. Type Checking / Semantic Analysis
    let mut tc = TypeChecker::new();
    tc.check_program(&program);

    // 4. ZK Circuit Generation Pass (Must handle constraint errors gracefully without panicking)
    let mut zk_emitter = ZkEmitter::new();
    let _ = zk_emitter.emit_program(&program);
});
