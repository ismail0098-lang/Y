//! Execute tagged enums against independent expected results at two LLVM levels.
//! Constructor arguments must run once, payload fields must keep their declared
//! types, and destructuring must select the active variant and preserve scope.
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

fn parse(source: &str) -> y::ast::Program {
    y::parser::Parser::new(y::lexer::Lexer::new(source).tokenize())
        .parse_program().expect("parse enum fixture")
}

fn run(source: &str, checks: &str) {
    let program = parse(source);
    let mut checker = y::type_checker::TypeChecker::new();
    checker.check_program(&program);
    assert!(checker.errors.is_empty(), "{:?}", checker.errors);
    let mut emitter = y::llvm_emitter::LlvmEmitter::new();
    let ir = emitter.emit_program(&program, &y::sentinel::HardwareProfile::default());
    assert!(emitter.emit_errors.is_empty(), "{:?}", emitter.emit_errors);
    if Command::new("clang").arg("--version").output().is_err() {
        eprintln!("SKIP enum execution: clang unavailable");
        return;
    }
    let dir = pinned::scratch("enum_payloads");
    let ll = dir.join("enum.ll");
    let c = dir.join("oracle.c");
    let bin = dir.join("enum_test");
    std::fs::write(&ll, &ir).unwrap();
    std::fs::write(&c, format!("#include <stdint.h>\n#include <assert.h>\n{checks}")).unwrap();
    for optimization in ["-O0", "-O2"] {
        let built = Command::new("clang").args([optimization, "-Wno-override-module"])
            .arg(&ll).arg(&c).arg("-o").arg(&bin).output().unwrap();
        assert!(built.status.success(), "{}\n{ir}", String::from_utf8_lossy(&built.stderr));
        let result = Command::new(&bin).output().unwrap();
        assert!(result.status.success(), "{optimization}: {:?} {}", result.status.code(), String::from_utf8_lossy(&result.stderr));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn all_eight_fields_keep_width_signedness_and_float_bits_through_copies_and_calls() {
    run(r#"
enum Value_Box { Empty, Pair(I8, U16, I64, F32, F64, bool, char, U32), Word(I64) }
fn construct() -> Value_Box {
    return Value_Box::Pair(-7, 65535, 9007199254740993, 1.25, 2.5, true, 'Z', 4294967295);
}
fn mutate(p: &mut Value_Box) { p.data.Pair._1 = 123; p.data.Pair._4 = 4.5; }
fn mixed() -> I32 {
    let mut p: Value_Box = construct();
    let copy: Value_Box = p;
    if copy.data.Pair._0 != -7 || copy.data.Pair._1 <= 32767 { return 0; }
    if copy.data.Pair._2 != 9007199254740993 { return 0; }
    if copy.data.Pair._3 != 1.25 || copy.data.Pair._4 != 2.5 { return 0; }
    if !copy.data.Pair._5 || copy.data.Pair._6 != 'Z' { return 0; }
    if copy.data.Pair._7 <= 2147483647 { return 0; }
    mutate(&mut p);
    if p.data.Pair._1 != 123 || p.data.Pair._4 != 4.5 { return 0; }
    if copy.data.Pair._1 != 65535 || copy.data.Pair._4 != 2.5 { return 0; }
    return 1;
}
"#, "extern int32_t mixed(void);\nint main(void) { assert(mixed() == 1); }");
}

#[test]
fn constructor_argument_side_effects_run_once_in_source_order() {
    run(r#"
enum Count_Pair { Pair(I32, I32) }
@unsafe
fn next(n: &mut I32) -> I32 { *n = *n + 1; return *n; }
fn once() -> I32 {
    let mut n: I32 = 0;
    let p = Count_Pair::Pair(next(&mut n), next(&mut n));
    if n != 2 || p.data.Pair._0 != 1 || p.data.Pair._1 != 2 { return 0; }
    return 1;
}
"#, "extern int32_t once(void);\nint main(void) { assert(once() == 1); }");
}

#[test]
fn matches_extract_typed_fields_and_keep_bindings_inside_each_arm() {
    run(r#"
enum Sum_Value { Zero, Many(I16, U32, F64), Other(I32) }
enum Color { Red, Blue }
@unsafe
fn record(out: &mut I32, n: I32) { *out = n; }
@unsafe
fn score(out: &mut I32, a: I16, b: U32, c: F64) {
    if a == -3 && b > 2147483647 && c == 1.5 { *out = 31; }
}
fn matched() -> I32 {
    let value = Sum_Value::Many(-3, 4294967295, 1.5);
    let mut result: I32 = 0;
    let a: I32 = 77;
    match value {
        Sum_Value::Other(v) => record(&mut result, 99),
        Many(a, b, c) => score(&mut result, a, b, c),
        _ => record(&mut result, 88)
    }
    if a != 77 || result != 31 { return 0; }
    match Sum_Value::Zero {
        Sum_Value::Many(a, b, c) => record(&mut result, 99),
        Sum_Value::Zero => record(&mut result, 41),
        _ => record(&mut result, 88)
    }
    if result != 41 { return 0; }
    match Color::Blue { Color::Red => record(&mut result, 99), Color::Blue => record(&mut result, 51) }
    if result != 51 { return 0; }
    match value { whole => score(&mut result, whole.data.Many._0, whole.data.Many._1, whole.data.Many._2) }
    return result;
}
"#, "extern int32_t matched(void);\nint main(void) { assert(matched() == 31); }");
}

#[test]
fn references_in_payloads_keep_their_pointee_width() {
    run(r#"
enum Ref_Value { Some(&mut I16, &mut I32) }
struct Pair { a: I16, b: I16, c: I32, d: I32 }
@unsafe
fn write(value: Ref_Value) { *value.data.Some._0 = -9; *value.data.Some._1 = 17; }
fn references() -> I32 {
    let mut p: Pair = Pair { a: 1, b: 2, c: 3, d: 4 };
    let value = Ref_Value::Some(&mut p.a, &mut p.c);
    write(value);
    if p.a != -9 || p.b != 2 || p.c != 17 || p.d != 4 { return 0; }
    return 1;
}
"#, "extern int32_t references(void);\nint main(void) { assert(references() == 1); }");
}

#[test]
fn unsupported_payload_layouts_are_refused_explicitly() {
    for (source, message) in [
        ("struct P { a: I32 } enum E { V(P) }", "supported scalar layout"),
        ("enum E { V([I32; 2]) }", "supported scalar layout"),
        ("enum E { V(Q16.16) }", "fixed-point payloads are unsupported"),
        ("enum E { V(I32,I32,I32,I32,I32,I32,I32,I32,I32) }", "at most 8"),
        ("enum E<T> { Unit }", "monomorphization"),
        ("enum Xa_Yb { Zc } enum Xa { Yb_Zc }", "ambiguous"),
    ] {
        let mut emitter = y::llvm_emitter::LlvmEmitter::new();
        emitter.emit_program(&parse(source), &y::sentinel::HardwareProfile::default());
        assert!(emitter.emit_errors.iter().any(|e| e.contains(message)), "{source}: {:?}", emitter.emit_errors);
    }
}

#[test]
fn invalid_fields_and_patterns_fail_before_codegen() {
    for (body, message) in [
        ("let v = E::Pair(1, 2); let x = v.data.Pair._2;", "has no field"),
        ("let v = E::Pair(1, 2); match v { E::Pair(a) => use_value(a) }", "expects 2 binding"),
        ("let v = E::Pair(1, 2); match v { E::Pair(a, a) => use_value(a) }", "duplicate enum match binding"),
        ("let v = E::Pair;", "requires constructor arguments"),
        ("let v = E::Pair(1, 2); match v { Other::Pair(a, b) => use_value(a) }", "does not name a variant"),
    ] {
        let source = format!("enum E {{ Pair(I32, I32) }} enum Other {{ Pair(I32, I32) }} fn use_value(x: I32) {{}} fn check() {{ {body} }}");
        let mut checker = y::type_checker::TypeChecker::new();
        checker.check_program(&parse(&source));
        assert!(checker.errors.iter().any(|e| e.contains(message)), "{source}: {:?}", checker.errors);
    }
}
