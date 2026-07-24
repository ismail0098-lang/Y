#![no_main]
use libfuzzer_sys::fuzz_target;
use y::lexer::Lexer;
use y::parser::Parser;

fuzz_target!(|data: &[u8]| {
    // Fuzzing inputs must be valid UTF-8 strings for the Y language lexer
    let source = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return,
    };

    // 1. Isolate and test Lexer
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();

    if tokens.is_empty() {
        return;
    }

    // 2. Isolate and test Parser
    let mut parser = Parser::new(tokens);
    let _ = parser.parse_program();
});
