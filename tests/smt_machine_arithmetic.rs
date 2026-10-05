//! Invariant proofs must describe machine arithmetic, including every
//! intermediate operation, assignment conversion, and loop condition.
use std::process::Command;
use std::sync::OnceLock;
use y::type_checker::{z3_candidates, TypeChecker};

#[path = "common/verification.rs"]
mod verification;

fn solver_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let found = z3_candidates().iter().any(|candidate| {
            Command::new(candidate)
                .arg("-version")
                .output()
                .is_ok_and(|out| out.status.success())
        });
        verification::prerequisite_available(found, "executable Z3 for SMT machine arithmetic tests")
    })
}

fn check(source: &str) -> Vec<String> {
    assert_ne!(
        std::env::var("Y_ALLOW_UNVERIFIED_INVARIANTS").as_deref(),
        Ok("1"),
        "these regressions require actual invariant verification"
    );
    let program = y::parser::Parser::new(y::lexer::Lexer::new(source).tokenize())
        .parse_program()
        .expect("parse arithmetic regression");
    let mut checker = TypeChecker::new();
    checker.check_program(&program);
    assert!(
        checker
            .errors
            .iter()
            .all(|e| !e.contains("SMT solver could not be run")),
        "a discovered Z3 must run; solver failure is not a successful rejection: {:?}",
        checker.errors
    );
    checker.errors
}

fn reject(label: &str, source: &str) {
    let errors = check(source);
    assert!(
        errors
            .iter()
            .any(|e| e.contains("SMT Safety Verification Failed")
                || e.contains("Cannot verify invariant")),
        "{label}: expected an invariant verification failure, got {errors:?}\n{source}"
    );
}

fn accept(label: &str, source: &str) {
    let errors = check(source);
    assert!(errors.is_empty(), "{label}: {errors:?}\n{source}");
}

#[test]
fn overflowing_addition_cannot_preserve_a_nonnegative_invariant() {
    if !solver_available() {
        return;
    }
    for (ty, maximum) in [("I32", "2147483647"), ("I64", "9223372036854775807")] {
        for update in ["x = x + 1;", "x += 1;"] {
            reject(
                ty,
                &format!(
                    "fn main() {{ let x: {ty} = {maximum};\n\
                 @invariant(x >= 0) for i in 0..1 {{ {update} }} }}"
                ),
            );
        }
    }
}

#[test]
fn negative_division_and_remainder_do_not_use_euclidean_smt_semantics() {
    if !solver_available() {
        return;
    }
    for ty in ["I32", "I64"] {
        for update in ["x = x / 2;", "x /= 2;"] {
            reject(
                "negative quotient",
                &format!(
                    "fn main() {{ let x: {ty} = -1;\n\
                 @invariant(x <= -1) for i in 0..1 {{ {update} }} }}"
                ),
            );
        }
        for update in ["x = (0 - 1) % 2;", "x = -1; x = x % 2;"] {
            reject(
                "negative remainder",
                &format!(
                    "fn main() {{ let x: {ty} = 0;\n\
                 @invariant(x >= 0) for i in 0..1 {{ {update} }} }}"
                ),
            );
        }
    }
}

fn division_program(name: &str, ty: &str, a: i64, b: i64, op: &str, compound: bool) -> String {
    let expected = if op == "/" { a / b } else { a % b };
    let update = if compound {
        format!("x {op}= {b};")
    } else {
        format!("x = x {op} {b};")
    };
    format!(
        "fn {name}() -> {ty} {{ let x: {ty} = {a};\n\
         @invariant((i == 0 && x == {a}) || (i == 1 && x == {expected}))\n\
         for i in 0..1 {{ {update} }} return x; }}\n"
    )
}

#[test]
fn truncating_division_and_signed_remainders_verify_for_every_sign_combination() {
    if !solver_available() {
        return;
    }
    for ty in ["I32", "I64"] {
        for (a, b) in [(-5, 2), (5, -2), (-5, -2), (5, 2)] {
            for op in ["/", "%"] {
                for compound in [false, true] {
                    // The surface parser has /= but no %= token.
                    if op == "%" && compound {
                        continue;
                    }
                    let source = division_program("calculate", ty, a, b, op, compound);
                    accept("truncation toward zero", &source);
                }
            }
        }
    }
}

#[test]
fn narrowing_a_wide_positive_result_cannot_prove_a_positive_i32() {
    if !solver_available() {
        return;
    }
    reject(
        "assignment narrowing",
        r#"
fn main() {
    let x: I32 = 0;
    let wide: I64 = 2147483647;
    @invariant(x >= 0)
    for i in 0..1 { x = wide + 1; }
}
"#,
    );
}

#[test]
fn intermediate_overflow_is_checked_before_division_or_widening() {
    if !solver_available() {
        return;
    }
    reject(
        "overflow before division",
        r#"
fn main() {
    let x: I32 = 2147483647;
    @invariant(x >= 0)
    for i in 0..1 { x = (x + 1) / 2; }
}
"#,
    );
    // The initializer runs at I32 width before conversion to I64. Its
    // mathematical value cannot be asserted as a trusted entry interval.
    reject(
        "overflow before widening",
        r#"
fn main() {
    let wide: I64 = 2147483647 + 1;
    @invariant(wide > 0)
    for i in 0..1 { }
}
"#,
    );
}

#[test]
fn postbody_assignment_intervals_do_not_make_preservation_vacuous() {
    if !solver_available() {
        return;
    }
    for loop_source in ["for i in 0..1 { x = -1; }", "while x >= 0 { x = -1; }"] {
        reject(
            "stale body facts",
            &format!("fn main() {{ let x: I32 = 0; @invariant(x >= 0) {loop_source} }}"),
        );
    }
}

#[test]
fn overflow_in_a_condition_cannot_hide_a_reachable_body() {
    if !solver_available() {
        return;
    }
    reject(
        "condition overflow",
        r#"
fn main() {
    let x: I32 = 2147483647;
    @invariant(x == 2147483647)
    while x + 1 < 0 { x = 0; }
}
"#,
    );
}

#[test]
fn invariant_expressions_themselves_must_have_defined_machine_arithmetic() {
    if !solver_available() {
        return;
    }
    reject(
        "overflow in invariant",
        r#"
fn main() {
    let x: I32 = 2147483647;
    @invariant(x + 1 > 0)
    for i in 0..1 { }
}
"#,
    );
    reject(
        "division by zero",
        r#"
fn main() {
    let x: I32 = 1;
    @invariant(x / 0 == x / 0)
    for i in 0..1 { }
}
"#,
    );
    reject(
        "signed division overflow",
        r#"
fn main() {
    let x: I32 = 0 - 2147483647 - 1;
    @invariant(x / -1 > 0)
    for i in 0..1 { }
}
"#,
    );
}

#[test]
fn accepted_signed_division_proofs_match_executed_llvm_results() {
    if !solver_available() {
        return;
    }
    let have_clang = Command::new("clang")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success());
    if !verification::prerequisite_available(have_clang, "clang to execute verified arithmetic") {
        return;
    }
    let mut source = String::new();
    let mut harness = String::from("#include <stdint.h>\n#include <assert.h>\n");
    let mut assertions = String::new();
    for (index, (a, b)) in [(-5, 2), (5, -2), (-5, -2), (5, 2)].into_iter().enumerate() {
        for (prefix, op, expected) in [("quotient", "/", a / b), ("remainder", "%", a % b)] {
            let name = format!("{prefix}_{index}");
            source.push_str(&division_program(&name, "I32", a, b, op, false));
            harness.push_str(&format!("extern int32_t {name}(void);\n"));
            assertions.push_str(&format!("assert({name}() == {expected});\n"));
        }
    }
    accept("executed arithmetic proofs", &source);
    let program = y::parser::Parser::new(y::lexer::Lexer::new(&source).tokenize())
        .parse_program()
        .unwrap();
    let mut emitter = y::llvm_emitter::LlvmEmitter::new();
    let ir = emitter.emit_program(&program, &y::sentinel::HardwareProfile::default());
    assert!(emitter.emit_errors.is_empty(), "{:?}", emitter.emit_errors);
    harness.push_str(&format!("int main(void) {{\n{assertions}return 0;\n}}\n"));
    let dir = std::env::temp_dir().join(format!("y_smt_machine_arithmetic_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ll = dir.join("verified.ll");
    let c = dir.join("harness.c");
    let exe = dir.join("run");
    std::fs::write(&ll, ir).unwrap();
    std::fs::write(&c, harness).unwrap();
    let output = Command::new("clang")
        .args(["-O2", "-Wno-override-module"])
        .arg(&ll)
        .arg(&c)
        .arg("-o")
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "clang: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(&exe).output().unwrap();
    assert!(
        output.status.success(),
        "verified division produced an incorrect value: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn comparisons_refuse_literals_with_different_backend_signedness() {
    if !solver_available() {
        return;
    }
    // LLVM promotes to I64 and says true. PTX promotes to U32 and compares
    // 0xffffffff < 0x80000000, which is false.
    reject(
        "mixed literal signedness",
        r#"
fn main() {
    let x: I32 = -1;
    @invariant(x < 2147483648)
    for i in 0..1 { }
}
"#,
    );
}

#[test]
fn unsigned_proofs_are_refused_until_backend_operators_agree() {
    if !solver_available() {
        return;
    }
    for ty in ["U32", "U64"] {
        reject(
            "unsupported unsigned arithmetic",
            &format!("fn main() {{ let x: {ty} = 7; @invariant(x >= 0) for i in 0..1 {{ }} }}"),
        );
    }
    // The same simple proof remains available for the supported signed type.
    accept(
        "signed control",
        r#"
fn main() {
    let x: I32 = 7;
    @invariant(x >= 0)
    for i in 0..1 { }
}
"#,
    );
}

#[test]
fn compound_division_cannot_assume_a_wider_rhs_than_llvm_uses() {
    if !solver_available() {
        return;
    }
    // LLVM emits sdiv i32 with this immediate, truncating it to one. PTX
    // promotes the division to 64 bits. Refuse a proof claiming their results
    // agree while the backends implement these different coercions.
    reject(
        "compound assignment width",
        r#"
fn main() {
    let x: I32 = 5;
    @invariant((i == 0 && x == 5) || (i == 1 && x == 0))
    for i in 0..1 { x /= 4294967297; }
}
"#,
    );
}

#[test]
fn entry_intervals_do_not_assume_signed_division_for_unsigned_literal_expressions() {
    if !solver_available() {
        return;
    }
    // Both intermediate mathematical values fit I32. PTX nevertheless
    // computes in U32 because 2147483648 is unsigned there, so its quotient
    // is positive. Merely checking the final interval's width is insufficient.
    reject(
        "entry interval signedness",
        r#"
fn main() {
    let wide: I64 = (0 - 2147483648) / 2;
    @invariant(wide < 0)
    for i in 0..1 { }
}
"#,
    );
}

#[test]
fn entry_intervals_do_not_treat_unsigned_intermediates_as_negative_wide_values() {
    if !solver_available() {
        return;
    }
    for expression in ["0 - 2147483648", "(0 - 2147483648) + 1", "-2147483648"] {
        reject(
            "unsigned intermediate widened to I64",
            &format!(
                "fn main() {{ let wide: I64 = {expression};\n\
             @invariant(wide < 0) for i in 0..1 {{ }} }}"
            ),
        );
    }
}

#[test]
fn a_negative_dynamic_step_cannot_reuse_an_increasing_loop_range() {
    if !solver_available() {
        return;
    }
    reject(
        "negative dynamic step",
        r#"
fn main() {
    let decrement: I32 = -1;
    @invariant(i >= 0)
    for i in 1..3 step decrement { }
}
"#,
    );
}

#[test]
fn changing_the_induction_variable_cannot_invalidate_the_assumed_lower_bound() {
    if !solver_available() {
        return;
    }
    // One step from i=1 produces zero and satisfies the written invariant.
    // The next iteration is still reachable and produces -1; assuming i>=1
    // on every iteration would incorrectly hide that second step.
    reject(
        "loop lower bound not preserved",
        r#"
fn main() {
    @invariant(i >= 0)
    for i in 1..3 { i = i - 2; }
}
"#,
    );
}

#[test]
fn a_shadowing_local_cannot_repair_the_tracked_outer_binding() {
    if !solver_available() {
        return;
    }
    let errors = check(
        r#"
fn main() {
    let x: I32 = 0;
    @invariant(x >= 0)
    for i in 0..1 {
        x = -1;
        @safe { let x: I32 = 0; }
    }
}
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("shadows an existing binding")),
        "a name-based SSA model must refuse lexical shadowing: {errors:?}"
    );
}
