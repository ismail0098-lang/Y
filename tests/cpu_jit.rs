//! Execute the general LLVM CPU JIT directly. Compilation failures are test
//! failures on its supported platform; these tests never skip missing LLVM.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use y::cpu_jit::{CpuJit, JitOptions};

fn compile(source: &str, opt_level: u8) -> CpuJit {
    CpuJit::compile_with_options(
        source,
        JitOptions {
            opt_level,
            ..JitOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("O{opt_level}: {error}\n{source}"))
}

fn rejected(source: &str) -> String {
    match CpuJit::compile(source) {
        Ok(_) => panic!("invalid source compiled:\n{source}"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn recursion_nested_control_flow_and_main_alias_execute_at_o0_and_o3() {
    let source = r#"
fn fib(n: I32) -> I32 {
    if n <= 0 { return 0; }
    if n == 1 { return 1; }
    return fib(n - 1) + fib(n - 2);
}

@unsafe
fn main() -> I32 {
    let mut total: I32 = 0;
    for i in 0..4 {
        let mut j: I32 = 0;
        while j < 3 {
            if (i + j) % 2 == 0 { total = total + i * 10 + j; }
            j = j + 1;
        }
    }
    return fib(12) + total;
}
"#;
    for opt in [0, 3] {
        let jit = compile(source, opt);
        assert_eq!(
            jit.function_address("main").unwrap(),
            jit.function_address("ysu_main").unwrap()
        );
        assert!(jit.functions().any(|name| name == "fib"));
        assert!(jit.optimized_ir().contains("target triple"));
        assert!(!jit.llvm_version().is_empty());
        unsafe {
            let fib: unsafe extern "C" fn(i32) -> i32 =
                std::mem::transmute(jit.function_address("fib").unwrap());
            assert_eq!(fib(0), 0);
            assert_eq!(fib(1), 1);
            assert_eq!(fib(12), 144);
            // The six even-parity terms are 0, 2, 11, 20, 22, and 31.
            assert_eq!(jit.run_main().unwrap(), 230);
        }
    }
}

#[test]
fn local_array_copies_assignments_and_by_value_parameters_do_not_alias() {
    let source = r#"
fn poke(a: [I32; 3]) -> I32 {
    a[0] = 99;
    return a[0] + a[2];
}
fn main() -> I32 {
    let mut v: [I32; 3] = {};
    v[0] = 7;
    v[2] = 11;
    let mut w: [I32; 3] = v;
    v[2] = 1;
    let mut u: [I32; 3] = {};
    u = w;
    w[0] = 0;
    let result: I32 = poke(u);
    return v[0] * 100 + u[0] * 10 + result + u[0];
}
"#;
    for opt in [0, 3] {
        let jit = compile(source, opt);
        assert_eq!(unsafe { jit.run_main().unwrap() }, 887);
    }
}

#[test]
fn host_memory_references_use_byte_word_and_integer_pointee_widths() {
    let source = r#"
@unsafe
fn bump16(a: &mut [I16; 4]) {
    a[1] = a[1] + 5;
    a[3] = a[3] + 9;
}
@unsafe
fn bytes(a: &mut [U8; 4]) {
    a[0] = 200;
    a[3] = 255;
}
@unsafe
fn set32(r: &mut I32) { *r = 7; }
@unsafe
fn set64(r: &mut I64) { *r = 8589934593; }
"#;
    for opt in [0, 3] {
        let jit = compile(source, opt);
        unsafe {
            let bump16: unsafe extern "C" fn(*mut i16) =
                std::mem::transmute(jit.function_address("bump16").unwrap());
            let bytes: unsafe extern "C" fn(*mut u8) =
                std::mem::transmute(jit.function_address("bytes").unwrap());
            let set32: unsafe extern "C" fn(*mut i32) =
                std::mem::transmute(jit.function_address("set32").unwrap());
            let set64: unsafe extern "C" fn(*mut i64) =
                std::mem::transmute(jit.function_address("set64").unwrap());
            let mut words = [1_i16, 2, 3, 4, 71, 72, 73, 74];
            bump16(words.as_mut_ptr());
            assert_eq!(words, [1, 7, 3, 13, 71, 72, 73, 74]);
            let mut octets = [1_u8, 2, 3, 4, 71, 72, 73, 74];
            bytes(octets.as_mut_ptr());
            assert_eq!(octets, [200, 2, 3, 255, 71, 72, 73, 74]);
            let mut integers = [1_i32, 0x12345678];
            set32(integers.as_mut_ptr());
            assert_eq!(integers, [7, 0x12345678]);
            let mut wide = 0_i64;
            set64(&mut wide);
            assert_eq!(wide, 8589934593);
        }
    }
}

#[test]
fn float_arguments_and_returns_preserve_f32_and_f64_precision() {
    let source = r#"
fn single(x: F32, y: F32) -> F32 { return x * y + 0.5; }
fn double(x: F64, y: F64) -> F64 { return x * y + 0.5; }
fn root(x: F32) -> F32 { return math_sqrt(x); }
"#;
    for opt in [0, 3] {
        let jit = compile(source, opt);
        unsafe {
            let single: unsafe extern "C" fn(f32, f32) -> f32 =
                std::mem::transmute(jit.function_address("single").unwrap());
            let double: unsafe extern "C" fn(f64, f64) -> f64 =
                std::mem::transmute(jit.function_address("double").unwrap());
            let root: unsafe extern "C" fn(f32) -> f32 =
                std::mem::transmute(jit.function_address("root").unwrap());
            assert_eq!(single(1.5, 2.25), 3.875);
            assert_eq!(double(1.0 + 2_f64.powi(-40), 2.0), 2.5 + 2_f64.powi(-39));
            assert_eq!(root(81.0), 9.0);
        }
    }
}

#[test]
fn returned_structs_with_array_fields_retain_their_contents() {
    let source = r#"
struct Box3 { v: [I32; 3], tag: I32, }
fn make() -> Box3 {
    let mut b: Box3 = {};
    b.v[1] = 17;
    b.v[2] = 29;
    b.tag = 5;
    return b;
}
fn main() -> I32 {
    let b: Box3 = make();
    return b.v[1] * 100 + b.v[2] + b.tag;
}
"#;
    for opt in [0, 3] {
        let jit = compile(source, opt);
        assert_eq!(unsafe { jit.run_main().unwrap() }, 1734);
    }
}

#[test]
fn string_construction_cloning_mutation_equality_and_queries_execute() {
    let source = r#"
fn main() -> I32 {
    let mut original: String = String_new("ab");
    let mut copy: String = String_clone(&original);
    String_push(&mut copy, 'c');
    String_push_str(&mut copy, &original);
    let mut answer: I32 = 0;
    if String_len(&original) == 2 && String_len(&copy) == 5 {
        if String_char_at(&copy, 2) == 'c' && String_eq_cstr(&copy, "abcab") {
            if !String_eq(&original, &copy) { answer = 57; }
        }
    }
    String_free(&mut original);
    String_free(&mut copy);
    return answer;
}
"#;
    for opt in [0, 3] {
        let jit = compile(source, opt);
        assert_eq!(unsafe { jit.run_main().unwrap() }, 57);
    }
}

#[test]
fn vector_char_push_get_and_length_execute() {
    let source = r#"
fn main() -> I32 {
    let mut v: Vec = Vec_new(1);
    let first: char = 'A';
    let second: char = 'z';
    Vec_push(&mut v, &first);
    Vec_push(&mut v, &second);
    let mut answer: I32 = 0;
    if Vec_len(&v) == 2 && Vec_get_char(&v, 0) == 'A' && Vec_get_char(&v, 1) == 'z' {
        answer = 91;
    }
    Vec_free(&mut v);
    return answer;
}
"#;
    for opt in [0, 3] {
        let jit = compile(source, opt);
        assert_eq!(unsafe { jit.run_main().unwrap() }, 91);
    }
}

#[test]
fn file_callbacks_write_read_and_preserve_content() {
    for opt in [0, 3] {
        let path =
            std::env::temp_dir().join(format!("y-cpu-jit-file-{}-O{opt}.txt", std::process::id()));
        let source = format!(
            r#"
fn main() -> I32 {{
    let mut path: String = "{}";
    let mut content: String = "hello from the JIT";
    File_write(&path, &content);
    let mut read: String = File_read_to_string(&path);
    let mut answer: I32 = 0;
    if String_eq(&read, &content) {{ answer = 19; }}
    String_free(&mut path);
    String_free(&mut content);
    String_free(&mut read);
    return answer;
}}
"#,
            path.display()
        );
        let jit = compile(&source, opt);
        assert_eq!(unsafe { jit.run_main().unwrap() }, 19);
        assert_eq!(std::fs::read(&path).unwrap(), b"hello from the JIT");
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn numeric_string_helpers_preserve_values_above_32_bits() {
    let source = r#"
fn main() -> I32 {
    let mut text: String = "8589934593";
    let parsed: I64 = str_to_i64(text);
    let mut answer: I32 = 0;
    if parsed == 8589934593 && ychar_to_ascii('z') == 122 { answer = 43; }
    String_free(&mut text);
    return answer;
}
"#;
    for opt in [0, 3] {
        let jit = compile(source, opt);
        assert_eq!(unsafe { jit.run_main().unwrap() }, 43);
    }
}

#[test]
fn rejects_invalid_semantics_unsupported_host_lowering_and_unresolved_symbols() {
    assert!(
        rejected("@unsafe\nfn main() -> I32 { let p: I32 = 1; return *p; }")
            .contains("Cannot dereference")
    );
    assert!(
        rejected("fn main() -> I32 { return definitely_not_a_y_function(); }")
            .contains("definitely_not_a_y_function")
    );
    let gpu = rejected("fn main() -> I32 { return thread_idx_x(); }");
    assert!(
        gpu.contains("LLVM host backend") && gpu.contains("thread_idx_x"),
        "{gpu}"
    );
    let array_return =
        rejected("fn make() -> [I32; 3] { let a: [I32; 3] = {}; return a; }\nfn main() {}");
    assert!(array_return.contains("returns an array"), "{array_return}");
    // This name is recognized as a runtime function by the frontend, but the
    // GUI runtime is absent. Compilation must fail naming the missing symbol.
    let external = rejected("fn main() -> I32 { return init_shadowplay_gui(); }");
    assert!(external.contains("init_shadowplay_gui"), "{external}");
    assert!(CpuJit::compile_with_options(
        "fn main() {}",
        JitOptions {
            opt_level: 4,
            ..JitOptions::default()
        }
    )
    .is_err());
}

#[test]
fn repeated_compilation_isolates_identical_symbols_and_releases_sessions() {
    for round in 0..8 {
        let first = compile(
            &format!("fn answer() -> I32 {{ return {}; }}", 100 + round),
            round % 2 * 3,
        );
        let second = compile(
            &format!("fn answer() -> I32 {{ return {}; }}", 200 + round),
            round % 2 * 3,
        );
        unsafe {
            let a: unsafe extern "C" fn() -> i32 =
                std::mem::transmute(first.function_address("answer").unwrap());
            let b: unsafe extern "C" fn() -> i32 =
                std::mem::transmute(second.function_address("answer").unwrap());
            assert_eq!(a(), 100 + i32::from(round));
            assert_eq!(b(), 200 + i32::from(round));
            drop(first);
            assert_eq!(b(), 200 + i32::from(round));
        }
        assert!(second.function_address("absent").is_err());
    }
}

#[test]
fn user_defined_runtime_names_take_precedence_and_void_main_runs() {
    let source = "fn String_len(s: &String) -> I64 { return 99; }\nfn main() {}";
    for opt in [0, 3] {
        let jit = compile(source, opt);
        unsafe {
            let length: unsafe extern "C" fn(*const std::ffi::c_void) -> i64 =
                std::mem::transmute(jit.function_address("String_len").unwrap());
            assert_eq!(length(std::ptr::null()), 99);
            assert_eq!(jit.run_main().unwrap(), 0);
        }
    }
}
