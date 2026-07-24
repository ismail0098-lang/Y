#![no_main]
use libfuzzer_sys::fuzz_target;
use y::lexer::Lexer;
use y::parser::Parser;
use y::sentinel::HardwareProfile;
use y::type_checker::TypeChecker;
use y::llvm_emitter::LlvmEmitter;

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
        Err(_) => return,
    };

    let mut tc = TypeChecker::new();
    tc.check_program(&program);

    let hw_profile = HardwareProfile::default();
    let mut llvm_emitter = LlvmEmitter::new();
    let _ = llvm_emitter.emit_program(&program, &hw_profile);
});
