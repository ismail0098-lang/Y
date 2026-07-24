#![no_main]
use libfuzzer_sys::fuzz_target;
use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;

fuzz_target!(|data: &[u8]| {
    let source = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();

    if tokens.is_empty() {
        return;
    }

    let mut parser = Parser::new(tokens);
    let program = match parser.parse_program() {
        Ok(p) => p,
        Err(_) => return, // Valid syntax errors are discarded
    };

    // Test semantic analysis / type checking pass
    let mut tc = TypeChecker::new();
    tc.check_program(&program);
});
