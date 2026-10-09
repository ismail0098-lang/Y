#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::ffi::c_void;
use y::cpu_jit::{CpuJit, JitOptions, JitValue};

const SOURCE: &str = r#"
@unsafe
fn string_scan(n: I64) -> I64 {
    let mut text: String = "";
    let mut i: I64 = 0;
    while i < n {
        if (i & 1) == 0 { String_push(&mut text, 'A'); }
        else { String_push(&mut text, 'z'); }
        i = i + 1;
    }
    let mut result: I64 = String_len(&text) * 1000;
    i = 0;
    while i < String_len(&text) {
        result = result + ychar_to_ascii(String_char_at(&text, i));
        i = i + 1;
    }
    result = result + ychar_to_ascii(String_char_at(&text, -1));
    result = result + ychar_to_ascii(String_char_at(&text, String_len(&text)));
    String_free(&mut text);
    result = result + String_len(&text) + ychar_to_ascii(String_char_at(&text, 0));
    String_free(&mut text);
    return result;
}
fn string_clone_bytes() -> I64 {
    let mut text: String = "aÈ";
    let mut copy: String = String_clone(&text);
    String_push(&mut copy, 'È');
    let result: I64 = String_len(&copy) * 1000000
        + ychar_to_ascii(String_char_at(&copy, 1)) * 10000
        + ychar_to_ascii(String_char_at(&copy, 2)) * 100
        + ychar_to_ascii(String_char_at(&copy, 3));
    String_free(&mut text);
    String_free(&mut copy);
    return result;
}
@unsafe
fn string_mutation_loop(n: I64) -> I64 {
    let mut text: String = "";
    let mut i: I64 = 0;
    let mut result: I64 = 0;
    while i < n {
        result = result + String_len(&text) * 100;
        String_push(&mut text, 'A');
        result = result + String_len(&text) * 10 + ychar_to_ascii(String_char_at(&text, i));
        i = i + 1;
    }
    String_free(&mut text);
    return result;
}
@unsafe
fn freed_slot_in_loop(n: I64) -> I64 {
    let mut text: String = "ab";
    let mut i: I64 = 0;
    let mut result: I64 = 0;
    while i < n {
        result = result + String_len(&text);
        String_free(&mut text);
        result = result + ychar_to_ascii(String_char_at(&text, 0));
        i = i + 1;
    }
    String_free(&mut text);
    return result;
}
@unsafe
fn vector_scan(n: I64) -> I64 {
    let mut values: Vec = Vec_new(1);
    let mut i: I64 = 0;
    while i < n {
        let mut value: char = 'z';
        if (i & 1) == 0 { value = 'A'; }
        Vec_push(&mut values, &value);
        i = i + 1;
    }
    let mut result: I64 = Vec_len(&values) * 1000;
    i = 0;
    while i < Vec_len(&values) {
        result = result + ychar_to_ascii(Vec_get_char(&values, i));
        i = i + 1;
    }
    result = result + ychar_to_ascii(Vec_get_char(&values, -1));
    result = result + ychar_to_ascii(Vec_get_char(&values, Vec_len(&values)));
    Vec_free(&mut values);
    result = result + Vec_len(&values) + ychar_to_ascii(Vec_get_char(&values, 0));
    return result;
}
@unsafe
fn vector_i64(n: I64, index: I64) -> I64 {
    let mut values: Vec = yvec_new(8);
    let mut i: I64 = 0;
    while i < n {
        let value: I64 = i * 4294967296 + 7;
        Vec_push(&mut values, &value);
        i = i + 1;
    }
    let mut result: I64 = 0;
    if index >= 0 && index < Vec_len(&values) {
        let loaded: I64 = load(yvec_get(&values, index));
        result = loaded;
    }
    Vec_free(&mut values);
    return result;
}
fn invalid_vector() -> I64 {
    let mut values: Vec = Vec_new(0);
    let result: I64 = Vec_len(&values) + ychar_to_ascii(Vec_get_char(&values, 0));
    Vec_free(&mut values);
    return result;
}
fn ascii(value: char) -> I32 { return ychar_to_ascii(value); }
fn equality_reference() -> bool {
    let mut text: String = "hello";
    let result: bool = String_eq_cstr(&text, &text);
    String_free(&mut text);
    return result;
}
fn equality_alias() -> bool {
    let mut text: String = "hello";
    let alias: &String = &text;
    let result: bool = String_eq_cstr(&text, alias);
    String_free(&mut text);
    return result;
}
fn make_string() -> String { return "abc"; }
fn make_short_string() -> String { return "a"; }
fn external_len(text: &String) -> I64 { return String_len(text); }
fn external_free(text: &mut String) { String_free(text); }
fn external_push(text: &mut String, value: char) { String_push(text, value); }
fn external_equal(a: &String, b: &String) -> bool { return String_eq_cstr(a, b); }
fn external_cstr(a: &String, b: &char) -> bool { return String_eq_cstr(a, b); }
fn aliased_slot() -> I64 {
    let mut text: String = "ab";
    let borrowed: &String = &text;
    let mut result: I64 = String_len(borrowed);
    String_free(&mut text);
    result = result + String_len(borrowed);
    return result;
}
fn copied_handle() -> I64 {
    let mut text: String = "ab";
    let mut copy: String = text;
    let mut result: I64 = String_len(&copy);
    String_free(&mut text);
    result = result + String_len(&copy);
    String_free(&mut copy);
    return result;
}
fn reassigned_handle() -> I64 {
    let mut text: String = "ab";
    String_free(&mut text);
    text = "xyz";
    let result: I64 = String_len(&text);
    String_free(&mut text);
    return result;
}
fn opaque_length(text: &String) -> I64 { return String_len(text); }
fn opaque_callee() -> I64 {
    let mut text: String = "ab";
    let result: I64 = opaque_length(&text);
    String_free(&mut text);
    return result;
}
"#;

fn compile(source: &str, level: u8, enabled: bool) -> CpuJit {
    compile_flags(source, level, enabled, false)
}

fn compile_flags(source: &str, level: u8, queries: bool, mutations: bool) -> CpuJit {
    compile_copy_flags(source, level, queries, mutations, false)
}

fn compile_copy_flags(
    source: &str,
    level: u8,
    queries: bool,
    mutations: bool,
    copies: bool,
) -> CpuJit {
    CpuJit::compile_with_options(
        source,
        JitOptions {
            opt_level: level,
            optimize_runtime: queries,
            optimize_runtime_mutations: mutations,
            optimize_runtime_copies: copies,
            ..JitOptions::default()
        },
    )
    .unwrap()
}

fn function_ir(jit: &CpuJit, name: &str) -> String {
    let lines: Vec<_> = jit.optimized_ir().lines().collect();
    let start = lines
        .iter()
        .position(|line| line.starts_with("define ") && line.contains(&format!("@{name}(")))
        .unwrap();
    lines[start..]
        .iter()
        .take_while(|line| **line != "}")
        .copied()
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn closed_local_reads_match_expected_values_with_growth_bounds_and_freed_slots() {
    for level in [0, 3] {
        for enabled in [false, true] {
            let jit = compile(SOURCE, level, enabled);
            for n in [-1, 0, 1, 8, 9, 65] {
                let count = n.max(0);
                let expected = count * 1000 + ((count + 1) / 2) * 65 + (count / 2) * 122;
                for function in ["string_scan", "vector_scan"] {
                    assert_eq!(
                        unsafe { jit.call(function, &[JitValue::I64(n)]).unwrap() },
                        JitValue::I64(expected),
                        "{function}, O{level}, enabled={enabled}, n={n}"
                    );
                }
                let expected_mutation = 110 * count * (count - 1) / 2 + 75 * count;
                assert_eq!(
                    unsafe {
                        jit.call("string_mutation_loop", &[JitValue::I64(n)])
                            .unwrap()
                    },
                    JitValue::I64(expected_mutation)
                );
                assert_eq!(
                    unsafe { jit.call("freed_slot_in_loop", &[JitValue::I64(n)]).unwrap() },
                    JitValue::I64(if n > 0 { 2 } else { 0 })
                );
                for index in [-1, 0, 7, 8, 64, 65] {
                    assert_eq!(
                        unsafe {
                            jit.call("vector_i64", &[JitValue::I64(n), JitValue::I64(index)])
                                .unwrap()
                        },
                        JitValue::I64(if index >= 0 && index < n {
                            index * 4294967296 + 7
                        } else {
                            0
                        })
                    );
                }
            }
            assert_eq!(
                unsafe { jit.call("string_clone_bytes", &[]).unwrap() },
                JitValue::I64(5_963_800)
            );
            assert_eq!(
                unsafe { jit.call("invalid_vector", &[]).unwrap() },
                JitValue::I64(0)
            );
            assert_eq!(
                unsafe { jit.call("equality_reference", &[]).unwrap() },
                JitValue::Bool(true)
            );
            assert_eq!(
                unsafe { jit.call("equality_alias", &[]).unwrap() },
                JitValue::Bool(true)
            );
            for byte in [0, 1, 127, 128, 200, 255] {
                assert_eq!(
                    unsafe { jit.call("ascii", &[JitValue::U8(byte)]).unwrap() },
                    JitValue::I32(i32::from(byte))
                );
            }
            for (name, result) in [
                ("aliased_slot", 2),
                ("copied_handle", 2),
                ("reassigned_handle", 3),
                ("opaque_callee", 2),
            ] {
                assert_eq!(
                    unsafe { jit.call(name, &[]).unwrap() },
                    JitValue::I64(result)
                );
            }
        }
    }
}

#[test]
fn only_proven_closed_local_queries_are_compiler_visible() {
    for level in [0, 3] {
        let fast = compile(SOURCE, level, true);
        let opaque = compile(SOURCE, level, false);
        for (name, queries) in [
            ("string_scan", &["String_len", "String_char_at"][..]),
            ("vector_scan", &["Vec_len", "Vec_get_char"][..]),
            ("vector_i64", &["Vec_len", "yvec_get"][..]),
        ] {
            let fast_ir = function_ir(&fast, name);
            let opaque_ir = function_ir(&opaque, name);
            for query in queries {
                assert!(
                    !fast_ir.contains(&format!("@{query}(")),
                    "O{level} {name}:\n{fast_ir}"
                );
                assert!(
                    opaque_ir.contains(&format!("@{query}(")),
                    "O{level} {name}:\n{opaque_ir}"
                );
            }
            assert!(!fast_ir.contains("@ychar_to_ascii("), "{fast_ir}");
            assert!(fast_ir.contains("_free("));
            assert!(fast_ir.contains("_push("));
        }
        for name in [
            "external_len",
            "aliased_slot",
            "copied_handle",
            "reassigned_handle",
            "opaque_callee",
        ] {
            let ir = function_ir(&fast, name);
            assert!(!ir.contains("CPU JIT runtime header read"), "{name}:\n{ir}");
            assert!(!ir.contains("runtime.read"), "{name}:\n{ir}");
        }
    }
}

#[test]
fn external_handles_and_reference_slots_retain_registry_resolution() {
    for enabled in [false, true] {
        let jit = compile(SOURCE, 3, enabled);
        unsafe {
            let make: unsafe extern "C" fn() -> *mut c_void =
                std::mem::transmute(jit.function_address("make_string").unwrap());
            let len: unsafe extern "C" fn(*const c_void) -> i64 =
                std::mem::transmute(jit.function_address("external_len").unwrap());
            let release: unsafe extern "C" fn(*mut c_void) =
                std::mem::transmute(jit.function_address("external_free").unwrap());
            let mut handle = make();
            assert!(!handle.is_null());
            assert_eq!(len(handle), 3);
            assert_eq!(len((&handle as *const *mut c_void).cast()), 3);
            assert_eq!(len(std::ptr::null()), 0);
            release((&mut handle as *mut *mut c_void).cast());
            assert!(handle.is_null());
            assert_eq!(len((&handle as *const *mut c_void).cast()), 0);
            release((&mut handle as *mut *mut c_void).cast());
        }
    }
}

#[test]
fn equality_normalization_preserves_typed_slots_direct_handles_binary_bytes_and_short_c_text() {
    for enabled in [false, true] {
        let jit = compile(SOURCE, 3, enabled);
        unsafe {
            let make: unsafe extern "C" fn() -> *mut c_void =
                std::mem::transmute(jit.function_address("make_string").unwrap());
            let make_short: unsafe extern "C" fn() -> *mut c_void =
                std::mem::transmute(jit.function_address("make_short_string").unwrap());
            let push: unsafe extern "C" fn(*mut c_void, u8) =
                std::mem::transmute(jit.function_address("external_push").unwrap());
            let equal: unsafe extern "C" fn(*const c_void, *const c_void) -> bool =
                std::mem::transmute(jit.function_address("external_equal").unwrap());
            let cstr: unsafe extern "C" fn(*const c_void, *const u8) -> bool =
                std::mem::transmute(jit.function_address("external_cstr").unwrap());
            let release: unsafe extern "C" fn(*mut c_void) =
                std::mem::transmute(jit.function_address("external_free").unwrap());
            let mut handle = make();
            push(handle, 0);
            push(handle, b'q');
            let slot = (&mut handle as *mut *mut c_void).cast::<c_void>();
            for first in [handle.cast_const(), slot.cast_const()] {
                for second in [handle.cast_const(), slot.cast_const()] {
                    assert!(
                        equal(first, second),
                        "direct/slot forms must preserve embedded NUL bytes"
                    );
                }
                assert!(
                    !cstr(first, c"abc".as_ptr().cast()),
                    "raw C text terminates before the stored NUL byte"
                );
            }
            let mut short = make_short();
            assert!(
                cstr(short, c"a".as_ptr().cast()),
                "two-byte C string must not be read as an eight-byte slot"
            );
            assert!(!cstr(short, c"".as_ptr().cast()));
            assert!(!equal(std::ptr::null(), short));
            assert!(!equal(short, std::ptr::null()));
            release(slot);
            assert!(handle.is_null());
            assert!(!equal(slot, slot));
            release((&mut short as *mut *mut c_void).cast());
        }
    }
}

#[test]
fn user_defined_runtime_aliases_override_queries_and_constructors() {
    let source = r#"
        fn String_len(value: &String) -> I32 { return 99; }
        fn Vec_len(value: &Vec) -> I16 { return 77; }
        fn Vec_get_char(value: &Vec, index: usize) -> char { return 'È'; }
        fn String_new(value: String) -> I32 { return 31; }
        fn ychar_to_ascii(value: char) -> I64 { return 1234567890123; }
        fn ascii_override() -> I64 { return ychar_to_ascii('A'); }
        fn main() -> I32 {
            let mut text: String = "ab";
            let mut values: Vec = Vec_new(1);
            let mut result: I32 = String_len(&text) + Vec_len(&values) + String_new("xy");
            if Vec_get_char(&values, 0) == 'È' { result = result + 200; }
            String_free(&mut text);
            Vec_free(&mut values);
            return result;
        }
    "#;
    for enabled in [false, true] {
        let jit = compile(source, 3, enabled);
        assert_eq!(unsafe { jit.run_main().unwrap() }, 407);
        assert_eq!(
            unsafe { jit.call("ychar_to_ascii", &[JitValue::U8(200)]).unwrap() },
            JitValue::I64(1234567890123)
        );
        assert_eq!(
            unsafe { jit.call("ascii_override", &[]).unwrap() },
            JitValue::I64(1234567890123)
        );
    }
}

#[test]
fn aot_emitter_keeps_runtime_optimization_disabled() {
    let ast = y::parser::Parser::new(
        y::lexer::Lexer::new(
            "fn answer() -> I64 { let mut s: String = \"ab\"; return String_len(&s); }",
        )
        .tokenize(),
    )
    .parse_program()
    .unwrap();
    let mut emitter = y::llvm_emitter::LlvmEmitter::new();
    let ir = emitter.emit_program(&ast, &y::cpu_jit::host_profile());
    assert!(ir.contains("call i64 @String_len("));
    assert!(!ir.contains("CPU JIT runtime header read"));
    assert!(!ir.contains("runtime.read"));
}

#[test]
fn statement_bearing_indices_are_refused_before_any_header_fast_path() {
    use y::ast::{Block, Expr, Item, Span, Stmt, UnaryOp};
    let mut ast = y::parser::Parser::new(y::lexer::Lexer::new(
        "fn answer() -> char { let mut text: String = \"ab\"; return String_char_at(&text, 0); }"
    ).tokenize()).parse_program().unwrap();
    let Item::Func(function) = &mut ast.items[0] else {
        unreachable!()
    };
    let Stmt::Return(Some(Expr::Call { args, .. }), _) = &mut function.body.stmts[1] else {
        unreachable!()
    };
    let span = Span { line: 1, col: 1 };
    args[1] = Expr::BlockExpr(
        Block {
            stmts: vec![
                Stmt::Expr(Expr::Call {
                    func: Box::new(Expr::Ident("String_free".into(), span.clone())),
                    args: vec![Expr::UnaryOp {
                        op: UnaryOp::Ref { mutable: true },
                        operand: Box::new(Expr::Ident("text".into(), span.clone())),
                        span: span.clone(),
                    }],
                    span: span.clone(),
                }),
                Stmt::Expr(Expr::IntLit(0, span.clone())),
            ],
            span: span.clone(),
        },
        span,
    );
    for enabled in [false, true] {
        let mut emitter = y::llvm_emitter::LlvmEmitter::new();
        emitter.register_host_runtime_symbols(&["ystr_new", "String_char_at", "String_free"]);
        emitter.set_optimize_runtime(enabled);
        let ir = emitter.emit_program(&ast, &y::cpu_jit::host_profile());
        assert!(
            !emitter.emit_errors.is_empty(),
            "statement-bearing arguments currently have no executable lowering"
        );
        assert!(!ir.contains("CPU JIT runtime header read"));
        assert!(!ir.contains("runtime.read"));
    }
}

const APPEND_SOURCE: &str = r#"
@unsafe
fn odd_sized_vector(n: I64) -> I64 {
    let mut values: Vec = Vec_new(3);
    let mut i: I64 = 0;
    while i < n {
        let value: I32 = 1122867;
        Vec_push(&mut values, &value);
        i = i + 1;
    }
    let mut result: I64 = Vec_len(&values) * 1000;
    i = 0;
    while i < Vec_len(&values) {
        result = result + ychar_to_ascii(Vec_get_char(&values, i));
        i = i + 1;
    }
    Vec_free(&mut values);
    return result;
}
@unsafe
fn self_source_growth() -> I64 {
    let mut values: Vec = yvec_new(8);
    let mut i: I64 = 0;
    while i < 8 {
        let value: I64 = 100 + i;
        yvec_push(&mut values, &value);
        i = i + 1;
    }
    yvec_push(&mut values, yvec_get(&values, 3));
    yvec_push(&mut values, yvec_get(&values, 0));
    let first: I64 = load(yvec_get(&values, 8));
    let second: I64 = load(yvec_get(&values, 9));
    Vec_free(&mut values);
    return first * 1000 + second;
}
fn null_and_freed_push() -> I64 {
    let mut text: String = "";
    String_push(&mut text, 'A');
    String_free(&mut text);
    ystr_push(&mut text, 'B');
    let mut values: Vec = Vec_new(1);
    let value: char = 'È';
    Vec_push(&mut values, &value);
    Vec_free(&mut values);
    Vec_push(&mut values, &value);
    let mut invalid: Vec = Vec_new(0);
    Vec_push(&mut invalid, &value);
    let result: I64 = String_len(&text) + Vec_len(&values) + Vec_len(&invalid);
    Vec_free(&mut invalid);
    return result;
}
fn pushed_zero_byte() -> I64 {
    let mut text: String = "abc";
    String_push(&mut text, '\0');
    String_push(&mut text, 'È');
    let result: I64 = String_len(&text) * 1000 + ychar_to_ascii(String_char_at(&text, 3)) * 10 + ychar_to_ascii(String_char_at(&text, 4));
    String_free(&mut text);
    return result;
}
@unsafe
fn dynamic_element_size(size: I32) -> I64 {
    let mut values: Vec = Vec_new(size);
    let value: I64 = 5;
    Vec_push(&mut values, &value);
    let result: I64 = Vec_len(&values);
    Vec_free(&mut values);
    return result;
}
@unsafe
fn implicit_scalar_address() -> I64 {
    let mut values: Vec = Vec_new(8);
    let value: I64 = 4294967313;
    let mut i: I64 = 0;
    while i < 9 {
        Vec_push(&mut values, value);
        i = i + 1;
    }
    let result: I64 = load(yvec_get(&values, 8));
    Vec_free(&mut values);
    return result;
}
"#;

#[test]
fn guarded_appends_match_capacity_boundary_values_with_each_flag_independent() {
    let source = format!("{SOURCE}\n{APPEND_SOURCE}");
    for level in [0, 3] {
        for queries in [false, true] {
            for mutations in [false, true] {
                let jit = compile_flags(&source, level, queries, mutations);
                for n in [
                    0, 1, 2, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                ] {
                    let expected = n * 1000 + ((n + 1) / 2) * 65 + (n / 2) * 122;
                    for name in ["string_scan", "vector_scan"] {
                        assert_eq!(
                            unsafe { jit.call(name, &[JitValue::I64(n)]).unwrap() },
                            JitValue::I64(expected),
                            "{name}, O{level}, queries={queries}, mutations={mutations}, n={n}"
                        );
                    }
                    assert_eq!(
                        unsafe { jit.call("odd_sized_vector", &[JitValue::I64(n)]).unwrap() },
                        JitValue::I64(n * 1051)
                    );
                    let index = n - 1;
                    assert_eq!(
                        unsafe {
                            jit.call("vector_i64", &[JitValue::I64(n), JitValue::I64(index)])
                                .unwrap()
                        },
                        JitValue::I64(if n == 0 { 0 } else { index * 4294967296 + 7 })
                    );
                }
                assert_eq!(
                    unsafe { jit.call("self_source_growth", &[]).unwrap() },
                    JitValue::I64(103100)
                );
                assert_eq!(
                    unsafe { jit.call("null_and_freed_push", &[]).unwrap() },
                    JitValue::I64(0)
                );
                assert_eq!(
                    unsafe { jit.call("pushed_zero_byte", &[]).unwrap() },
                    JitValue::I64(5200)
                );
                assert_eq!(
                    unsafe { jit.call("implicit_scalar_address", &[]).unwrap() },
                    JitValue::I64(4294967313)
                );
                for size in [0, 1, 3, 8] {
                    assert_eq!(
                        unsafe {
                            jit.call("dynamic_element_size", &[JitValue::I32(size)])
                                .unwrap()
                        },
                        JitValue::I64(if size == 0 { 0 } else { 1 })
                    );
                }
            }
        }
    }
}

#[test]
fn append_ir_has_capacity_guards_and_overlap_safe_constant_copies_with_callback_fallbacks() {
    let source = format!("{SOURCE}\n{APPEND_SOURCE}");
    let fast = compile_flags(&source, 0, false, true);
    let opaque = compile_flags(&source, 0, false, false);
    for name in [
        "string_scan",
        "vector_scan",
        "vector_i64",
        "odd_sized_vector",
        "self_source_growth",
        "implicit_scalar_address",
    ] {
        let optimized = function_ir(&fast, name);
        assert!(
            optimized.contains("runtime.append.fast"),
            "{name}:\n{optimized}"
        );
        assert!(
            optimized.contains("runtime.append.slow"),
            "{name}:\n{optimized}"
        );
        assert!(
            optimized.contains("_push("),
            "growth must keep callbacks: {optimized}"
        );
        assert!(!function_ir(&opaque, name).contains("runtime.append.fast"));
    }
    let string = function_ir(&fast, "string_scan");
    assert!(
        string.contains("store i8 0"),
        "fast append retains trailing NUL"
    );
    assert!(string.contains("sub i64") && string.contains("icmp sle i64"));
    let odd = function_ir(&fast, "odd_sized_vector");
    assert!(odd.contains("@llvm.memmove.p0.p0.i64(ptr align 1"));
    assert!(odd.contains("i64 3, i1 false"));
    assert!(!function_ir(&fast, "dynamic_element_size").contains("runtime.append.fast"));
    // Both unproved self-source pointers retain the original callback even
    // when one call has spare capacity and the other triggers realloc.
    let own_data = function_ir(&fast, "self_source_growth");
    assert_eq!(own_data.matches("call void @yvec_push(").count(), 3);
}

#[test]
fn mutation_overrides_and_handle_aliases_keep_source_calls() {
    let source = r#"
        fn String_push(value: &mut String, byte: char) -> I32 { return 19; }
        fn Vec_push(value: &mut Vec, byte: &char) -> I16 { return 23; }
        fn main() -> I32 {
            let mut text: String = "";
            let mut values: Vec = Vec_new(1);
            let byte: char = 'A';
            let first: I32 = String_push(&mut text, byte);
            let second: I16 = Vec_push(&mut values, &byte);
            String_free(&mut text);
            Vec_free(&mut values);
            return first + second;
        }
    "#;
    for level in [0, 3] {
        for mutations in [false, true] {
            let jit = compile_flags(source, level, true, mutations);
            assert_eq!(unsafe { jit.run_main().unwrap() }, 42);
            assert!(!jit.optimized_ir().contains("runtime.append.fast"));
        }
    }
    let raw_overrides = r#"
        fn ystr_push(value: &mut String, byte: char) -> I32 { return 29; }
        fn yvec_push(value: &mut Vec, byte: &char) -> I64 { return 31; }
        fn main() -> I32 {
            let mut text: String = "";
            let mut values: Vec = Vec_new(1);
            let byte: char = 'A';
            let result: I64 = ystr_push(&mut text, byte) + yvec_push(&mut values, &byte);
            String_free(&mut text);
            Vec_free(&mut values);
            return result;
        }
    "#;
    for level in [0, 3] {
        for mutations in [false, true] {
            let jit = compile_flags(raw_overrides, level, true, mutations);
            assert_eq!(unsafe { jit.run_main().unwrap() }, 60);
            assert!(!jit.optimized_ir().contains("runtime.append.fast"));
        }
    }
    let aliases = r#"
        fn main() -> I32 {
            let mut text: String = "a";
            let alias: &mut String = &mut text;
            String_push(alias, 'b');
            let result: I32 = String_len(&text);
            String_free(&mut text);
            return result;
        }
    "#;
    let jit = compile_flags(aliases, 0, true, true);
    assert_eq!(unsafe { jit.run_main().unwrap() }, 2);
    assert!(!jit.optimized_ir().contains("runtime.append.fast"));
}

const COPY_SOURCE: &str = r#"
@unsafe
fn dynamic_records(size: I32, n: I64) -> I64 {
    let mut values: Vec = Vec_new(size);
    let mut i: I64 = 0;
    while i < n {
        let value: I64 = 72623859790382856 + i * 4294967296;
        Vec_push(&mut values, &value);
        i = i + 1;
    }
    let mut result: I64 = Vec_len(&values) * 100000;
    i = 0;
    while i < Vec_len(&values) {
        let bytes: &[char; 8] = yvec_get(&values, i);
        let mut byte: I32 = 0;
        while byte < size {
            result = result + ychar_to_ascii(bytes[byte]) * (byte + 1) * (i + 1);
            byte = byte + 1;
        }
        i = i + 1;
    }
    Vec_free(&mut values);
    return result;
}
@unsafe
fn borrowed_dynamic_source(size: I32, source: &I64) -> I64 {
    let mut values: Vec = Vec_new(size);
    yvec_push(&mut values, source);
    let result: I64 = Vec_len(&values);
    Vec_free(&mut values);
    return result;
}
@unsafe
fn bulk_scan(n: I64, style: I32) -> I64 {
    let mut text: String = "";
    let mut chunk: String = "a";
    String_push(&mut chunk, '\0');
    String_push(&mut chunk, 'È');
    let mut i: I64 = 0;
    while i < n {
        if style == 0 { String_push_str(&mut text, &chunk); }
        else if style == 1 { ystr_push_str(text, chunk); }
        else { String_push_str(&mut text, &mut chunk); }
        i = i + 1;
    }
    let mut result: I64 = String_len(&text) * 100000;
    i = 0;
    while i < String_len(&text) {
        result = result + ychar_to_ascii(String_char_at(&text, i)) * (i + 1);
        i = i + 1;
    }
    String_free(&mut text);
    String_free(&mut chunk);
    return result;
}
@unsafe
fn bulk_self(n: I64) -> I64 {
    let mut text: String = "a";
    String_push(&mut text, 'B');
    let mut i: I64 = 0;
    while i < n {
        String_push_str(&mut text, &text);
        i = i + 1;
    }
    let mut result: I64 = String_len(&text) * 100000;
    i = 0;
    while i < String_len(&text) {
        result = result + ychar_to_ascii(String_char_at(&text, i)) * (i + 1);
        i = i + 1;
    }
    String_free(&mut text);
    return result;
}
fn bulk_empty_and_freed() -> I64 {
    let mut text: String = "x";
    let mut empty: String = "";
    let mut other: String = "yz";
    String_push_str(&mut text, &empty);
    String_push_str(&mut empty, &empty);
    String_free(&mut other);
    String_push_str(&mut text, &other);
    let result: I64 = String_len(&text) + String_len(&empty);
    String_free(&mut text);
    String_push_str(&mut text, &empty);
    String_free(&mut empty);
    String_push_str(&mut text, &empty);
    return result + String_len(&text) + String_len(&empty);
}
fn bulk_alias() -> I64 {
    let mut text: String = "x";
    let mut other: String = "yz";
    let alias: &String = &other;
    String_push_str(&mut text, alias);
    let result: I64 = String_len(&text);
    String_free(&mut text);
    String_free(&mut other);
    return result;
}
fn external_bulk(text: &mut String, other: &String) { String_push_str(text, other); }
"#;

fn copied_source() -> String {
    let mut source = format!("{SOURCE}\n{APPEND_SOURCE}\n{COPY_SOURCE}");
    for (width, ty, value) in [
        (1, "char", "'È'"),
        (2, "I16", "23100"),
        (4, "I32", "16909060"),
        (8, "I64", "72623859790382856"),
    ] {
        source.push_str(&format!(
            r#"
            @unsafe
            fn dynamic_width_{width}(size: I32, n: I64) -> I64 {{
                let mut values: Vec = Vec_new(size);
                let value: {ty} = {value};
                let mut i: I64 = 0;
                while i < n {{ yvec_push(&mut values, &value); i = i + 1; }}
                let mut result: I64 = Vec_len(&values) * 100000;
                i = 0;
                while i < Vec_len(&values) {{
                    let bytes: &[char; 8] = yvec_get(&values, i);
                    let mut byte: I32 = 0;
                    while byte < size {{
                        result = result + ychar_to_ascii(bytes[byte]) * (byte + 1) * (i + 1);
                        byte = byte + 1;
                    }}
                    i = i + 1;
                }}
                Vec_free(&mut values);
                return result;
            }}
        "#
        ));
    }
    source
}

fn expected_bytes(records: impl IntoIterator<Item = Vec<u8>>) -> i64 {
    let mut result = 0;
    for (record, bytes) in records.into_iter().enumerate() {
        result += 100000;
        for (index, byte) in bytes.into_iter().enumerate() {
            result += i64::from(byte) * (index as i64 + 1) * (record as i64 + 1);
        }
    }
    result
}

fn expected_string(bytes: &[u8]) -> i64 {
    bytes.len() as i64 * 100000
        + bytes
            .iter()
            .enumerate()
            .map(|(i, byte)| i64::from(*byte) * (i as i64 + 1))
            .sum::<i64>()
}

#[test]
fn exact_width_dynamic_copies_and_bulk_appends_match_every_byte_at_o0_and_o3() {
    let source = copied_source();
    for level in [0, 3] {
        for queries in [false, true] {
            for mutations in [false, true] {
                for copies in [false, true] {
                    let jit = compile_copy_flags(&source, level, queries, mutations, copies);
                    for n in [0, 1, 2, 7, 8, 9, 15, 16, 17, 33] {
                        // All tested widths fit in the eight initialized source
                        // bytes. Only runtime width8 takes the exact-width path.
                        for size in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8] {
                            let expected = if size <= 0 {
                                0
                            } else {
                                expected_bytes((0..n).map(|i| {
                                    (72623859790382856_u64 + i as u64 * 4294967296).to_le_bytes()
                                        [..size as usize]
                                        .to_vec()
                                }))
                            };
                            assert_eq!(
                            unsafe {
                                jit.call(
                                    "dynamic_records",
                                    &[JitValue::I32(size), JitValue::I64(n)],
                                )
                                .unwrap()
                            },
                            JitValue::I64(expected),
                            "O{level}, queries={queries}, mutations={mutations}, copies={copies}, width={size}, n={n}"
                        );
                        }
                        for (width, bytes) in [
                            (1, vec![200]),
                            (2, 23100_i16.to_le_bytes().to_vec()),
                            (4, 16909060_i32.to_le_bytes().to_vec()),
                            (8, 72623859790382856_i64.to_le_bytes().to_vec()),
                        ] {
                            assert_eq!(
                                unsafe {
                                    jit.call(
                                        &format!("dynamic_width_{width}"),
                                        &[JitValue::I32(width), JitValue::I64(n)],
                                    )
                                    .unwrap()
                                },
                                JitValue::I64(expected_bytes((0..n).map(|_| bytes.clone())))
                            );
                        }
                        let text: Vec<u8> = (0..n).flat_map(|_| [b'a', 0, 200]).collect();
                        for style in [0, 1, 2] {
                            assert_eq!(
                                unsafe {
                                    jit.call("bulk_scan", &[JitValue::I64(n), JitValue::I32(style)])
                                        .unwrap()
                                },
                                JitValue::I64(expected_string(&text))
                            );
                        }
                    }
                    for n in 0..=6 {
                        let text: Vec<u8> = (0..(1 << n)).flat_map(|_| [b'a', b'B']).collect();
                        assert_eq!(
                            unsafe { jit.call("bulk_self", &[JitValue::I64(n)]).unwrap() },
                            JitValue::I64(expected_string(&text))
                        );
                    }
                    assert_eq!(
                        unsafe { jit.call("bulk_empty_and_freed", &[]).unwrap() },
                        JitValue::I64(1)
                    );
                    assert_eq!(
                        unsafe { jit.call("bulk_alias", &[]).unwrap() },
                        JitValue::I64(3)
                    );
                }
            }
        }
    }
}

#[test]
fn copy_ir_requires_both_flags_exact_scalar_extent_and_two_proven_string_handles() {
    let source = copied_source();
    let fast = compile_copy_flags(&source, 0, false, true, true);
    let old = compile_copy_flags(&source, 0, false, true, false);
    let no_mutations = compile_copy_flags(&source, 0, false, false, true);
    for width in [1, 2, 4, 8] {
        let name = format!("dynamic_width_{width}");
        let ir = function_ir(&fast, &name);
        assert!(ir.contains("runtime.append.fast"));
        assert!(
            ir.contains(&format!("i64 {width}, i1 false")),
            "fixed-width source copy: {ir}"
        );
        assert!(
            ir.contains("icmp eq i64"),
            "actual element width is checked: {ir}"
        );
        assert!(!function_ir(&old, &name).contains("runtime.append.fast"));
        assert!(!function_ir(&no_mutations, &name).contains("runtime.append.fast"));
    }
    assert!(function_ir(&fast, "odd_sized_vector").contains("i64 3, i1 false"));
    assert!(!function_ir(&fast, "borrowed_dynamic_source").contains("runtime.append.fast"));
    for name in ["bulk_scan", "bulk_self", "bulk_empty_and_freed"] {
        let ir = function_ir(&fast, name);
        assert!(ir.contains("runtime.copy.fast") && ir.contains("runtime.copy.slow"));
        assert!(ir.contains("@llvm.memmove.p0.p0.i64(ptr align 1"));
        assert!(ir.contains("store i8 0") && ir.contains("sub i64"));
        assert!(ir.contains("_push_str("), "growth retains callback: {ir}");
        assert!(!function_ir(&old, name).contains("runtime.copy.fast"));
        assert!(!function_ir(&no_mutations, name).contains("runtime.copy.fast"));
    }
    for name in ["bulk_alias", "external_bulk"] {
        assert!(!function_ir(&fast, name).contains("runtime.copy.fast"));
    }
}

#[test]
fn bulk_source_overrides_and_external_handle_forms_preserve_callbacks() {
    let overrides = r#"
        fn String_push_str(a: &mut String, b: &String) -> I16 { return 23; }
        fn ystr_push_str(a: &mut String, b: &String) -> I64 { return 31; }
        fn main() -> I32 {
            let mut text: String = "a";
            let mut other: String = "b";
            let result: I64 = String_push_str(&mut text, &other) + ystr_push_str(&mut text, &other);
            String_free(&mut text);
            String_free(&mut other);
            return result;
        }
    "#;
    for level in [0, 3] {
        for copies in [false, true] {
            let jit = compile_copy_flags(overrides, level, true, true, copies);
            assert_eq!(unsafe { jit.run_main().unwrap() }, 54);
            assert!(!jit.optimized_ir().contains("runtime.copy.fast"));
            let jit = compile_copy_flags(&copied_source(), level, true, true, copies);
            for destination_slot in [false, true] {
                for source_slot in [false, true] {
                    let JitValue::Pointer(mut destination) =
                        (unsafe { jit.call("make_short_string", &[]).unwrap() })
                    else {
                        unreachable!()
                    };
                    let JitValue::Pointer(mut source) =
                        (unsafe { jit.call("make_string", &[]).unwrap() })
                    else {
                        unreachable!()
                    };
                    let destination_arg = if destination_slot {
                        (&mut destination as *mut *mut c_void).cast()
                    } else {
                        destination
                    };
                    let source_arg = if source_slot {
                        (&mut source as *mut *mut c_void).cast()
                    } else {
                        source
                    };
                    assert_eq!(
                        unsafe {
                            jit.call(
                                "external_bulk",
                                &[
                                    JitValue::Pointer(destination_arg),
                                    JitValue::Pointer(source_arg),
                                ],
                            )
                            .unwrap()
                        },
                        JitValue::Void
                    );
                    assert_eq!(
                        unsafe {
                            jit.call("external_len", &[JitValue::Pointer(destination_arg)])
                                .unwrap()
                        },
                        JitValue::I64(4)
                    );
                    unsafe {
                        jit.call("external_free", &[JitValue::Pointer(destination_arg)])
                            .unwrap();
                        jit.call("external_free", &[JitValue::Pointer(source_arg)])
                            .unwrap();
                    }
                }
            }
        }
    }
}
