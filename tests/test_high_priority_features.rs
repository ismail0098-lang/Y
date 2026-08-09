// ============================================================
//  Integration Test Suite — High Priority Features & Debug Mode
// ============================================================

use y::ptx_emitter::PtxEmitter;
use y::sentinel::HardwareProfile;
use y::lexer::Lexer;
use y::parser::Parser;
use y::type_checker::TypeChecker;
use y::cpu_emitter::CpuEmitter;

#[test]
fn test_nd_broadcasting_emission() {
    let hw = HardwareProfile::default();
    let mut emitter = PtxEmitter::new_with_profile(&hw);
    let dst = emitter.emit_broadcast_to("%f0", &[1, 1024], &[128, 1024]);

    assert!(emitter.ptx_buffer.contains("N-D TENSOR BROADCASTING"));
    assert!(!dst.is_empty());
}

#[test]
fn test_cpu_interpreter_mode() {
    let src = r#"
        kernel vector_add(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>) {
            @invariant(i >= 0)
            for i in 0..100 step 1 {
                C[i] = A[i] + B[i];
            }
        }
    "#;

    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize();
    let mut parser = Parser::new(tokens);
    let ast = parser.parse_program().unwrap();

    let mut type_checker = TypeChecker::new();
    type_checker.check_program(&ast);
    assert!(type_checker.errors.is_empty(), "TypeChecker errors: {:?}", type_checker.errors);

    let mut cpu_emitter = CpuEmitter::new();
    let code = cpu_emitter.emit_program(&ast);
    assert!(code.contains("pub unsafe fn vector_add"));
}
