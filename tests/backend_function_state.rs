//! A function's lowering must not depend on unrelated functions before it.
use std::process::Command;

fn parse(source: &str) -> y::ast::Program {
    let tokens = y::lexer::Lexer::new(source).tokenize();
    y::parser::Parser::new(tokens)
        .parse_program()
        .expect("parse regression")
}

fn llvm(source: &str) -> (String, Vec<String>) {
    let mut emitter = y::llvm_emitter::LlvmEmitter::new();
    let ir = emitter.emit_program(&parse(source), &y::sentinel::HardwareProfile::default());
    (ir, emitter.emit_errors)
}

fn definition<'a>(module: &'a str, marker: &str) -> &'a str {
    let start = module.find(marker).expect("definition exists");
    let end = module[start..].find("\n}").expect("definition ends") + start + 2;
    &module[start..end]
}

fn clang_run(tag: &str, ir: &str, harness: &str) {
    if Command::new("clang").arg("--version").output().is_err() {
        eprintln!("SKIP execution: clang is absent; independent-emission checks passed");
        return;
    }
    let dir = std::env::temp_dir().join(format!("y_scope_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ll = dir.join("module.ll");
    let c = dir.join("harness.c");
    let bin = dir.join("run");
    std::fs::write(&ll, ir).unwrap();
    std::fs::write(&c, harness).unwrap();
    let out = Command::new("clang")
        .args(["-O0", "-Wno-override-module"])
        .arg(&ll)
        .arg(&c)
        .arg("-o")
        .arg(&bin)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        Command::new(&bin).status().unwrap().success(),
        "{tag} returned a wrong value"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn llvm_directives_are_local_to_functions_and_kernels() {
    let exact = [
        "fn preceding() -> I64 { @ZeroDrift let acc: I64 = 7; return acc; }",
        "kernel preceding() { @ZeroDrift let acc: I64 = 7; }",
    ];
    let plain = [
        ("fn ordinary() -> F32 { let mut acc: F32 = 1.5; acc = 7.5; return acc; }",
         "define float @ordinary(",
         "extern float ordinary(void); int main(void) { return ordinary() != 7.5f; }"),
        ("kernel ordinary(out: GlobalMemory<F32>) { let mut acc: F32 = 1.5; acc = 7.5; block_ptr2d_store(out, 0, 0, 1, 1, 1, acc); }",
         "define void @ordinary(",
         "extern void ordinary(float *); int main(void) { float out = 0; ordinary(&out); return out != 7.5f; }"),
    ];
    for (i, preceding) in exact.iter().enumerate() {
        for (j, (ordinary, marker, harness)) in plain.iter().enumerate() {
            let (standalone, errors) = llvm(ordinary);
            assert!(errors.is_empty(), "{errors:?}");
            let (combined, errors) = llvm(&format!("{preceding}\n{ordinary}"));
            assert!(
                errors.is_empty(),
                "state leaked across definitions: {errors:?}"
            );
            clang_run(&format!("drift_{i}_{j}"), &combined, harness);
            assert_eq!(
                definition(&combined, marker),
                definition(&standalone, marker)
            );
        }
    }
}

#[test]
fn llvm_inferred_locals_do_not_inherit_old_signedness() {
    let ordinary = "fn ordinary() -> I64 { let x = -7; return x; }";
    let (standalone, errors) = llvm(ordinary);
    assert!(errors.is_empty(), "{errors:?}");
    let (combined, errors) = llvm(&format!(
        "fn preceding(x: U32) -> U32 {{ return x; }}\n{ordinary}"
    ));
    assert!(errors.is_empty(), "{errors:?}");
    clang_run(
        "signedness",
        &combined,
        "extern long long ordinary(void); int main(void) { return ordinary() != -7; }",
    );
    assert_eq!(
        definition(&combined, "define i64 @ordinary("),
        definition(&standalone, "define i64 @ordinary(")
    );
}

#[test]
fn llvm_buffer_element_facts_do_not_escape_their_function() {
    let ordinary =
        "fn ordinary(data: I64) -> I64 { return block_ptr2d_load(data, 0, 0, 1, 1, 1); }";
    let (_, standalone_errors) = llvm(ordinary);
    assert!(
        !standalone_errors.is_empty(),
        "a scalar is not a typed buffer"
    );
    let (_, combined_errors) = llvm(&format!(
        "kernel preceding(data: GlobalMemory<I64>) {{}}\n{ordinary}"
    ));
    assert_eq!(
        combined_errors, standalone_errors,
        "the earlier pointer must not license a later scalar as a buffer"
    );
}

fn ptx(source: &str) -> String {
    let mut profile = y::sentinel::HardwareProfile::default();
    profile.sm_version = "80".into();
    let mut emitter = y::ptx_emitter::PtxEmitter::new_with_profile(&profile);
    let output = emitter.emit_program(&parse(source), &profile);
    assert!(emitter.emit_errors.is_empty(), "{:?}", emitter.emit_errors);
    output
}

fn assemble(tag: &str, module: &str) {
    if Command::new("ptxas").arg("--version").output().is_err() {
        eprintln!("SKIP assembly: ptxas is absent; independent-emission checks passed");
        return;
    }
    let dir = std::env::temp_dir().join(format!("y_scope_ptx_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("module.ptx");
    std::fs::write(&path, module).unwrap();
    let out = Command::new("ptxas")
        .args(["-arch=sm_80"])
        .arg(&path)
        .arg("-o")
        .arg(dir.join("module.cubin"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn ptx_directives_are_local_to_each_kernel() {
    let preceding = "kernel preceding() { @ZeroDrift let acc: I64 = 7; }";
    let ordinary = "kernel ordinary(out: GlobalMemory<F32>) { let mut acc: F32 = 1.5; acc = 7.5; store(out, 0, acc); }";
    let standalone = ptx(ordinary);
    let combined = ptx(&format!("{preceding}\n{ordinary}"));
    assert_eq!(
        definition(&combined, ".visible .entry ordinary("),
        definition(&standalone, ".visible .entry ordinary(")
    );
    assemble("directives", &combined);
}

#[test]
fn ptx_double_registers_start_fresh_in_each_kernel() {
    // General F64 values are not a supported PTX scalar type. The supported
    // fixed-point conversion path allocates actual double scratch registers.
    // With no branch in this fixture, label numbering cannot mask that check.
    let body = "(out: GlobalMemory<F32>, x: F32) { @bounds(min=0, max=1000) @ZeroDrift let mut acc: F32 = 0.0; acc += x; store(out, 0, acc); }";
    let ordinary = format!("kernel ordinary{body}");
    let standalone = ptx(&ordinary);
    let combined = ptx(&format!("kernel preceding{body}\n{ordinary}"));
    let standalone_definition = definition(&standalone, ".visible .entry ordinary(");
    assert!(
        standalone_definition.contains(".reg .f64 %fd<"),
        "fixture must allocate actual double scratch registers:\n{standalone_definition}"
    );
    assert!(
        standalone_definition.contains("cvt.f64.f32 %fd0,"),
        "the first conversion must start at double register zero:\n{standalone_definition}"
    );
    assert!(
        !standalone_definition.contains("bra "),
        "double-register fixture must not exercise label numbering"
    );
    assemble("double", &combined);
    assert_eq!(
        definition(&combined, ".visible .entry ordinary("),
        standalone_definition,
        "double scratch registers must start fresh in the second kernel"
    );
}

#[test]
fn ptx_labels_start_fresh_in_each_kernel() {
    let body = "(out: GlobalMemory<F32>, x: F32) { let mut acc: F32 = x; if x > 0.0 { acc = x + 1.0; } store(out, 0, acc); }";
    let ordinary = format!("kernel ordinary{body}");
    let standalone = ptx(&ordinary);
    let combined = ptx(&format!("kernel preceding{body}\n{ordinary}"));
    let standalone_definition = definition(&standalone, ".visible .entry ordinary(");
    assert!(
        standalone_definition.contains("bra "),
        "fixture must allocate branch labels:\n{standalone_definition}"
    );
    assert!(
        !standalone_definition.contains(".reg .f64"),
        "label fixture must not exercise double-register numbering"
    );
    assemble("labels", &combined);
    assert_eq!(
        definition(&combined, ".visible .entry ordinary("),
        standalone_definition,
        "branch labels must start fresh in the second kernel"
    );
}
