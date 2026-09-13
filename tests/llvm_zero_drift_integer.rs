//! Execute exact integer accumulators against independent C expectations.
//! Comparing two Y values could hide the same rounding bug on both sides.
use std::process::Command;

fn assert_integer_domain(ir: &str) {
    for conversion in ["sitofp", "uitofp", "fptosi", "fptoui"] {
        assert!(
            !ir.contains(conversion),
            "integer accumulator used {conversion}:\n{ir}"
        );
    }
}

fn run_case(name: &str, source: &str, harness: &str) {
    let tokens = y::lexer::Lexer::new(source).tokenize();
    let program = y::parser::Parser::new(tokens)
        .parse_program()
        .expect("parse regression source");
    let mut emitter = y::llvm_emitter::LlvmEmitter::new();
    let ir = emitter.emit_program(&program, &y::sentinel::HardwareProfile::default());
    assert!(emitter.emit_errors.is_empty(), "{:?}", emitter.emit_errors);
    if Command::new("clang").arg("--version").output().is_err() {
        // Keep a structural check even on machines without clang.
        assert_integer_domain(&ir);
        eprintln!("SKIP execution: clang is absent; integer-domain checks passed");
        return;
    }
    let dir = std::env::temp_dir().join(format!("y_drift_i64_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ll = dir.join("test.ll");
    let c = dir.join("harness.c");
    let bin = dir.join("run");
    std::fs::write(&ll, &ir).unwrap();
    std::fs::write(&c, harness).unwrap();
    for optimization in ["-O0", "-O2"] {
        let result = Command::new("clang")
            .args([optimization, "-Wno-override-module"])
            .arg(&ll)
            .arg(&c)
            .arg("-o")
            .arg(&bin)
            .output()
            .expect("run clang");
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let result = Command::new(&bin).output().expect("run regression harness");
        assert!(
            result.status.success(),
            "{name} {optimization}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    // Execute the independent C oracle first, so a before/after audit exposes
    // incorrect observable results rather than stopping at an IR difference.
    assert_integer_domain(&ir);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn initialization_and_readback_preserve_the_full_signed_integer() {
    run_case(
        "literals",
        r#"
fn above() -> I64 { @ZeroDrift let acc: I64 = 9007199254740993; return acc; }
fn below() -> I64 { @ZeroDrift let acc: I64 = 9007199254740991; return acc; }
fn negative() -> I64 { @ZeroDrift let acc: I64 = -9007199254740993; return acc; }
fn maximum() -> I64 { @ZeroDrift let acc: I64 = 9223372036854775807; return acc; }
fn near_minimum() -> I64 { @ZeroDrift let acc: I64 = -9223372036854775807; return acc; }
fn from_param(x: I64) -> I64 { @ZeroDrift let acc: I64 = x; return acc; }
fn empty() -> I64 { @ZeroDrift let acc: I64; return acc; }
"#,
        r#"
#include <stdint.h>
#include <assert.h>
extern int64_t above(void), below(void), negative(void), maximum(void), near_minimum(void), empty(void);
extern int64_t from_param(int64_t);
int main(void) {
    assert(above() == INT64_C(9007199254740993));
    assert(below() == INT64_C(9007199254740991));
    assert(negative() == -INT64_C(9007199254740993));
    assert(maximum() == INT64_MAX);
    assert(near_minimum() == -INT64_MAX);
    assert(from_param(INT64_MIN) == INT64_MIN);
    assert(from_param(INT64_MAX) == INT64_MAX);
    assert(empty() == 0);
}
"#,
    );
}

#[test]
fn every_accumulation_spelling_preserves_wide_terms_and_readers() {
    run_case(
        "updates",
        r#"
fn compound_add(x: I64) -> I64 { @ZeroDrift let mut acc: I64 = 0; acc += x; return acc; }
fn compound_sub(x: I64) -> I64 { @ZeroDrift let mut acc: I64 = 0; acc -= x; return acc; }
fn running_add(x: I64) -> I64 { @ZeroDrift let mut acc: I64 = 0; acc = acc + x; return acc; }
fn running_sub(x: I64) -> I64 { @ZeroDrift let mut acc: I64 = 0; acc = acc - x; return acc; }
fn read_expression(x: I64) -> I64 { @ZeroDrift let mut acc: I64 = x; acc += 1; return acc - x; }
fn read_comparison(x: I64) -> bool { @ZeroDrift let acc: I64 = x; return acc == x; }
fn copy_exact(x: I64) -> I64 { @ZeroDrift let first: I64 = x; @ZeroDrift let second: I64 = first; return second; }
"#,
        r#"
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
extern int64_t compound_add(int64_t), compound_sub(int64_t), running_add(int64_t), running_sub(int64_t);
extern int64_t read_expression(int64_t), copy_exact(int64_t);
extern bool read_comparison(int64_t);
int main(void) {
    int64_t values[] = {INT64_C(9007199254740991), INT64_C(9007199254740992),
        INT64_C(9007199254740993), -INT64_C(9007199254740991),
        -INT64_C(9007199254740992), -INT64_C(9007199254740993)};
    for (unsigned i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        int64_t v = values[i];
        assert(compound_add(v) == v);
        assert(compound_sub(v) == -v);
        assert(running_add(v) == v);
        assert(running_sub(v) == -v);
        assert(read_expression(v) == 1);
        assert(read_comparison(v));
        assert(copy_exact(v) == v);
    }
}
"#,
    );
}

#[test]
fn integer_terms_keep_signedness_when_widened() {
    run_case(
        "widen",
        r#"
fn initial_signed(x: I32) -> I64 { @ZeroDrift let acc: I64 = x; return acc; }
fn initial_unsigned(x: U32) -> I64 { @ZeroDrift let acc: I64 = x; return acc; }
fn add_signed(x: I32) -> I64 { @ZeroDrift let mut acc: I64 = 9007199254740993; acc += x; return acc; }
fn add_unsigned(x: U32) -> I64 { @ZeroDrift let mut acc: I64 = 9007199254740993; acc = acc + x; return acc; }
"#,
        r#"
#include <stdint.h>
#include <assert.h>
extern int64_t initial_signed(int32_t), initial_unsigned(uint32_t), add_signed(int32_t), add_unsigned(uint32_t);
int main(void) {
    assert(initial_signed(-7) == -7);
    assert(initial_unsigned(UINT32_MAX) == INT64_C(4294967295));
    assert(add_signed(-7) == INT64_C(9007199254740986));
    assert(add_unsigned(UINT32_MAX) == INT64_C(9007203549708288));
}
"#,
    );
}
