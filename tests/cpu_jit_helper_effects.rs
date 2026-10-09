#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use y::cpu_jit::{CpuJit, JitOptions, JitValue};

const SOURCE: &str = r#"
fn a_outer(x: I64) -> I64 { return a_middle(x) + z_leaf(2); }
fn a_middle(x: I64) -> I64 { return z_leaf(x) - 1; }
fn z_leaf(x: I64) -> I64 { return x * 3 + 7; }
fn upper(i: I64, seed: I64) -> bool { return ((i + seed) & 7) < 3; }
fn weight(i: I64) -> I64 { let result = (i & 7) + 1; return result; }
fn identity(x: I64) -> I64 { return x; }
fn ordered(a: I64, b: I64) -> I64 { return a * 100 + b; }
fn scalar_loop(n: I64) -> I64 {
    let mut result: I64 = 0;
    let mut i: I64 = 0;
    @invariant(i >= 0)
    while i < n { result += i; i += 1; }
    @invariant(j >= 0)
    for j in 0..3 { result += j; }
    return result;
}
fn scalar_shadow(x: I64) -> I64 {
    let mut result: I64 = x;
    if x >= 0 {
        let x: I64 = x + 2;
        @safe { let x: I64 = x + 3; result += x; }
        result += x;
    }
    return result;
}
fn scalar_f64(x: F64) -> F64 { return x * 1.5 + 0.25; }
@unsafe
fn scan(n: I64, seed: I64, element_size: I32) -> I64 {
    let mut text: String = "";
    let mut chunk: String = "xy";
    let mut values: Vec = Vec_new(element_size);
    let mut i: I64 = 0;
    while i < n {
        let value: I64 = a_outer(i + seed);
        Vec_push(&mut values, &value);
        if upper(i, seed) { String_push(&mut text, 'A'); }
        else { String_push(&mut text, 'z'); }
        i += 1;
    }
    String_push_str(&mut text, &chunk);
    let mut result: I64 = String_len(&text) * 10000 + Vec_len(&values) * 1000 + scalar_loop(n);
    i = 0;
    while i < Vec_len(&values) {
        let value: I64 = load(yvec_get(&values, i));
        result += value * weight(i);
        i += 1;
    }
    i = 0;
    while i < String_len(&text) {
        result += ychar_to_ascii(String_char_at(&text, i)) * weight(i);
        i += 1;
    }
    String_free(&mut text);
    String_free(&mut chunk);
    Vec_free(&mut values);
    return result;
}
fn recursive(n: I64) -> I64 {
    if n <= 0 { return 0; }
    return n + recursive(n - 1);
}
fn mutual_a(n: I64) -> I64 { if n <= 0 { return 0; } return 1 + mutual_b(n - 1); }
fn mutual_b(n: I64) -> I64 { if n <= 0 { return 0; } return 1 + mutual_a(n - 1); }
fn allocation_effect(x: I64) -> I64 {
    let mut hidden: String = "hidden";
    String_free(&mut hidden);
    return x + 5;
}
fn drop_string(text: &mut String) -> I64 { String_free(text); return 9; }
fn consume_string(text: &mut String) -> I64 {
    let length: I64 = String_len(text);
    String_free(text);
    return length;
}
@unsafe
fn read_pointer(value: &I64) -> I64 { return *value; }
fn native_conversion(value: char) -> I32 { return ychar_to_ascii(value); }
@unsafe
fn unsafe_scalar(value: I64) -> I64 { return value + 1; }
fn no_return_annotation(value: I64) { let ignored: I64 = value + 1; }
fn recursive_barrier(n: I64) -> I64 {
    let mut text: String = "a";
    let result: I64 = recursive(n);
    String_push(&mut text, 'B');
    let answer: I64 = result * 100 + String_len(&text);
    String_free(&mut text);
    return answer;
}
fn mutual_barrier(n: I64) -> I64 {
    let mut text: String = "a";
    let result: I64 = mutual_a(n);
    let answer: I64 = result * 100 + String_len(&text);
    String_free(&mut text);
    return answer;
}
fn allocation_barrier(n: I64) -> I64 {
    let mut text: String = "a";
    let result: I64 = allocation_effect(n);
    let answer: I64 = result * 100 + String_len(&text);
    String_free(&mut text);
    return answer;
}
fn pointer_barrier(n: I64) -> I64 {
    let mut text: String = "a";
    let result: I64 = read_pointer(&n);
    let answer: I64 = result * 100 + String_len(&text);
    String_free(&mut text);
    return answer;
}
fn nested_free_barrier() -> I64 {
    let mut text: String = "a";
    let result: I64 = identity(drop_string(&mut text));
    String_push(&mut text, 'B');
    let answer: I64 = result * 100 + String_len(&text);
    String_free(&mut text);
    return answer;
}
fn ordered_free_barrier() -> I64 {
    let mut text: String = "abc";
    return ordered(consume_string(&mut text), consume_string(&mut text));
}
fn runtime_barrier() -> I64 {
    let mut text: String = "a";
    let result: I64 = native_conversion('È');
    let answer: I64 = result * 100 + String_len(&text);
    String_free(&mut text);
    return answer;
}
fn annotation_barrier(n: I64) -> I64 {
    let mut text: String = "a";
    let result: I64 = unsafe_scalar(n);
    no_return_annotation(n);
    let answer: I64 = result * 100 + String_len(&text);
    String_free(&mut text);
    return answer;
}
fn profiled(n: I64, release: bool) -> I64 {
    let mut text: String = "a";
    let result: I64 = a_outer(n);
    if release { String_free(&mut text); }
    else { String_push(&mut text, 'B'); }
    let answer: I64 = result * 100 + String_len(&text);
    String_free(&mut text);
    return answer;
}
"#;

fn options(level: u8, helpers: bool, queries: bool, mutations: bool, copies: bool) -> JitOptions {
    JitOptions {
        opt_level: level,
        optimize_helper_effects: helpers,
        optimize_runtime: queries,
        optimize_runtime_mutations: mutations,
        optimize_runtime_copies: copies,
        ..JitOptions::default()
    }
}

fn function_ir(jit: &CpuJit, name: &str) -> String {
    let marker = format!("@{name}(");
    let start = jit
        .optimized_ir()
        .lines()
        .position(|line| line.starts_with("define ") && line.contains(&marker))
        .unwrap();
    jit.optimized_ir()
        .lines()
        .skip(start)
        .take_while(|line| *line != "}")
        .collect::<Vec<_>>()
        .join("\n")
}

fn expected_scan(n: i64, seed: i64) -> i64 {
    let count = n.max(0);
    let mut text: Vec<u8> = (0..count)
        .map(|i| if ((i + seed) & 7) < 3 { b'A' } else { b'z' })
        .collect();
    text.extend_from_slice(b"xy");
    let mut result = text.len() as i64 * 10000 + count * 1000 + count * (count - 1) / 2 + 3;
    result += (0..count)
        .map(|i| (3 * (i + seed) + 19) * ((i & 7) + 1))
        .sum::<i64>();
    result
        + text
            .iter()
            .enumerate()
            .map(|(i, byte)| i64::from(*byte) * (((i as i64) & 7) + 1))
            .sum::<i64>()
}

#[test]
fn direct_transitive_and_scoped_scalar_helpers_keep_runtime_paths_with_independent_flags() {
    for level in [0, 3] {
        for helpers in [false, true] {
            for queries in [false, true] {
                for mutations in [false, true] {
                    for copies in [false, true] {
                        let jit = CpuJit::compile_with_options(
                            SOURCE,
                            options(level, helpers, queries, mutations, copies),
                        )
                        .unwrap();
                        for (n, seed) in [
                            (-2, -4),
                            (0, 0),
                            (1, 7),
                            (7, -3),
                            (8, 2),
                            (9, 0),
                            (17, 3),
                            (33, -5),
                        ] {
                            assert_eq!(unsafe { jit.call("scan", &[JitValue::I64(n), JitValue::I64(seed), JitValue::I32(8)]).unwrap() }, JitValue::I64(expected_scan(n, seed)), "O{level}, helpers={helpers}, queries={queries}, mutations={mutations}, copies={copies}");
                        }
                        for x in [-3, 0, 5] {
                            assert_eq!(
                                unsafe { jit.call("a_outer", &[JitValue::I64(x)]).unwrap() },
                                JitValue::I64(3 * x + 19)
                            );
                            assert_eq!(
                                unsafe { jit.call("scalar_shadow", &[JitValue::I64(x)]).unwrap() },
                                JitValue::I64(if x < 0 { x } else { 3 * x + 7 })
                            );
                            assert_eq!(
                                unsafe {
                                    jit.call("scalar_f64", &[JitValue::F64(x as f64)]).unwrap()
                                },
                                JitValue::F64(x as f64 * 1.5 + 0.25)
                            );
                        }
                        if level == 0 {
                            let ir = function_ir(&jit, "scan");
                            assert_eq!(ir.contains("runtime.read"), helpers && queries);
                            assert_eq!(ir.contains("runtime.append.fast"), helpers && mutations);
                            assert_eq!(
                                ir.contains("runtime.copy.fast"),
                                helpers && mutations && copies
                            );
                            assert!(
                                ir.contains("call i64 @a_outer("),
                                "effect proof must retain source call evaluation"
                            );
                            assert!(ir.contains("_free("));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn impure_pointer_runtime_annotation_and_recursive_calls_keep_ownership_barriers() {
    for level in [0, 3] {
        for helpers in [false, true] {
            let jit =
                CpuJit::compile_with_options(SOURCE, options(level, helpers, true, true, true))
                    .unwrap();
            for (name, args, expected) in [
                ("recursive_barrier", vec![JitValue::I64(5)], 1502),
                ("mutual_barrier", vec![JitValue::I64(5)], 501),
                ("allocation_barrier", vec![JitValue::I64(5)], 1001),
                ("pointer_barrier", vec![JitValue::I64(5)], 501),
                ("nested_free_barrier", vec![], 900),
                ("ordered_free_barrier", vec![], 300),
                ("runtime_barrier", vec![], 20001),
                ("annotation_barrier", vec![JitValue::I64(5)], 601),
            ] {
                assert_eq!(
                    unsafe { jit.call(name, &args).unwrap() },
                    JitValue::I64(expected)
                );
                let ir = function_ir(&jit, name);
                assert!(
                    !ir.contains("runtime.read")
                        && !ir.contains("runtime.append.fast")
                        && !ir.contains("runtime.copy.fast"),
                    "{name}:\n{ir}"
                );
                if name != "ordered_free_barrier" || level == 3 {
                    assert!(
                        ir.contains("@String_len("),
                        "barrier retains callback: {ir}"
                    );
                } else {
                    assert_eq!(ir.matches("call i64 @consume_string(").count(), 2);
                }
            }
        }
    }
}

#[test]
fn proved_helpers_preserve_unseen_profiled_freeing_paths_and_option_identity() {
    for level in [0, 3] {
        let opt = options(level, true, true, true, true);
        let training = CpuJit::compile_instrumented(SOURCE, opt).unwrap();
        for _ in 0..30 {
            assert_eq!(
                unsafe {
                    training
                        .call("profiled", &[JitValue::I64(3), JitValue::Bool(false)])
                        .unwrap()
                },
                JitValue::I64(2802)
            );
        }
        let profile = training.branch_profile().unwrap();
        let final_jit = CpuJit::compile_with_profile(SOURCE, opt, &profile).unwrap();
        assert!(!final_jit.optimized_ir().contains("atomicrmw"));
        for n in [-4, 0, 3, 9] {
            for release in [false, true] {
                assert_eq!(
                    unsafe {
                        final_jit
                            .call("profiled", &[JitValue::I64(n), JitValue::Bool(release)])
                            .unwrap()
                    },
                    JitValue::I64((3 * n + 19) * 100 + if release { 0 } else { 2 })
                );
            }
        }
        assert!(CpuJit::compile_with_profile(
            SOURCE,
            JitOptions {
                optimize_helper_effects: false,
                ..opt
            },
            &profile
        )
        .err()
        .unwrap()
        .to_string()
        .contains("profile"));
        drop(training);
        assert_eq!(
            unsafe {
                final_jit
                    .call("profiled", &[JitValue::I64(3), JitValue::Bool(true)])
                    .unwrap()
            },
            JitValue::I64(2800)
        );
    }
}

#[test]
fn source_conversion_overrides_remain_source_calls_with_proved_scalar_effects() {
    let source = r#"
        fn ychar_to_ascii(value: char) -> I64 { return 1000; }
        fn answer() -> I64 {
            let mut text: String = "a";
            String_push(&mut text, 'B');
            let result: I64 = String_len(&text) * 100 + ychar_to_ascii(String_char_at(&text, 0));
            String_free(&mut text);
            return result;
        }
    "#;
    for level in [0, 3] {
        for helpers in [false, true] {
            let jit =
                CpuJit::compile_with_options(source, options(level, helpers, true, true, true))
                    .unwrap();
            assert_eq!(
                unsafe { jit.call("answer", &[]).unwrap() },
                JitValue::I64(1200)
            );
            if level == 0 {
                let ir = function_ir(&jit, "answer");
                assert!(ir.contains("call i64 @ychar_to_ascii("));
                assert_eq!(ir.contains("runtime.read"), helpers);
            }
        }
    }
}

#[test]
fn malformed_override_arguments_and_emitted_name_collisions_cannot_gain_runtime_protection() {
    use y::ast::{Block, Expr, ImplBlock, Item, KernelDecl, Program, Span, Stmt, UnaryOp};
    let parse = |source: &str| {
        y::parser::Parser::new(y::lexer::Lexer::new(source).tokenize())
            .parse_program()
            .unwrap()
    };
    let emit = |program: &Program| {
        let mut emitter = y::llvm_emitter::LlvmEmitter::new();
        emitter.register_host_runtime_symbols(&[
            "ystr_new",
            "ystr_len",
            "String_free",
            "String_len",
        ]);
        emitter.set_optimize_runtime(true);
        emitter.set_optimize_helper_effects(true);
        emitter.emit_program(program, &y::cpu_jit::host_profile())
    };
    let mut override_ast = parse(
        r#"
        fn String_len(value: I64) -> I64 { return 9; }
        fn driver() -> I64 {
            let mut text: String = "a";
            let ignored: I64 = String_len(0);
            let result: I64 = ystr_len(&text);
            String_free(&mut text);
            return result;
        }
    "#,
    );
    assert!(emit(&override_ast).contains("runtime.read"));
    let span = Span { line: 1, col: 1 };
    let Item::Func(driver) = &mut override_ast.items[1] else {
        unreachable!()
    };
    let Stmt::Let {
        init: Some(Expr::Call { args, .. }),
        ..
    } = &mut driver.body.stmts[1]
    else {
        unreachable!()
    };
    args[0] = Expr::UnaryOp {
        op: UnaryOp::Ref { mutable: true },
        operand: Box::new(Expr::Ident("text".into(), span.clone())),
        span: span.clone(),
    };
    // Deliberately bypass semantic checking to exercise the emitter's proof:
    // a scalar source override cannot borrow a host callback's object role.
    // The resulting invalid native call is never compiled or executed.
    assert!(!emit(&override_ast).contains("runtime.read"));

    let collision_source = r#"
        fn Helper_mix(value: I64) -> I64 { return value + 1; }
        fn driver() -> I64 {
            let mut text: String = "a";
            let value: I64 = Helper_mix(1);
            let result: I64 = value + ystr_len(&text);
            String_free(&mut text);
            return result;
        }
    "#;
    let ast = parse(collision_source);
    assert!(emit(&ast).contains("runtime.read"));
    let Item::Func(mut method) = ast.items[0].clone() else {
        unreachable!()
    };
    method.name = "mix".into();
    let mut method_collision = ast.clone();
    method_collision.items.push(Item::Impl(ImplBlock {
        target_type: "Helper".into(),
        generic_params: vec![],
        methods: vec![method],
        span: span.clone(),
    }));
    assert!(!emit(&method_collision).contains("runtime.read"));
    let mut kernel_collision = ast;
    kernel_collision.items.push(Item::Kernel(KernelDecl {
        requires: vec![],
        name: "Helper_mix".into(),
        params: vec![],
        body: Block {
            stmts: vec![],
            span: span.clone(),
        },
        tile: None,
        span,
    }));
    assert!(!emit(&kernel_collision).contains("runtime.read"));
}

#[test]
fn special_intrinsic_store_names_cannot_authorize_scalar_helper_effects() {
    let source = r#"
        fn block_ptr2d_store(a: I64, b: I64, c: I64, d: I64, e: I64, f: I64, g: I64) -> I64 {
            return g;
        }
        fn driver(buffer: GlobalMemory<I64>) -> I64 {
            let mut text: String = "a";
            let result: I64 = block_ptr2d_store(buffer, 0, 0, 1, 1, 1, 9);
            let answer: I64 = result * 100 + String_len(&text);
            String_free(&mut text);
            return answer;
        }
    "#;
    for level in [0, 3] {
        for helpers in [false, true] {
            let jit =
                CpuJit::compile_with_options(source, options(level, helpers, true, true, true))
                    .unwrap();
            let mut buffer = [3_i64, 17];
            // This records the existing intrinsic dispatch, which writes the
            // buffer and returns zero instead of calling the scalar source
            // definition. It does not endorse source override semantics.
            assert_eq!(
                unsafe {
                    jit.call("driver", &[JitValue::Pointer(buffer.as_mut_ptr().cast())])
                        .unwrap()
                },
                JitValue::I64(1)
            );
            assert_eq!(buffer, [9, 17]);
            let ir = function_ir(&jit, "driver");
            assert!(
                ir.contains("@String_len("),
                "O{level}, helpers={helpers}: {ir}"
            );
            assert!(
                !ir.contains("runtime.read")
                    && !ir.contains("runtime.append.fast")
                    && !ir.contains("runtime.copy.fast"),
                "intrinsic dispatch is an ownership barrier: O{level}, helpers={helpers}: {ir}"
            );
        }
    }
}

#[test]
fn ast_only_load_source_collision_cannot_authorize_scalar_helper_effects() {
    use y::ast::{Expr, Item, Stmt};
    let mut ast = y::parser::Parser::new(
        y::lexer::Lexer::new(
            r#"
                fn scalar_load(value: I64) -> I64 { return value + 1; }
                fn driver() -> I64 {
                    let mut text: String = "a";
                    let ignored: I64 = scalar_load(1);
                    let result: I64 = ystr_len(&text);
                    String_free(&mut text);
                    return result;
                }
            "#,
        )
        .tokenize(),
    )
    .parse_program()
    .unwrap();
    let Item::Func(helper) = &mut ast.items[0] else {
        unreachable!()
    };
    helper.name = "load".into();
    let Item::Func(driver) = &mut ast.items[1] else {
        unreachable!()
    };
    let Stmt::Let {
        init: Some(Expr::Call { func, .. }),
        ..
    } = &mut driver.body.stmts[1]
    else {
        unreachable!()
    };
    let Expr::Ident(name, _) = &mut **func else {
        unreachable!()
    };
    *name = "load".into();
    let mut emitter = y::llvm_emitter::LlvmEmitter::new();
    emitter.register_host_runtime_symbols(&["ystr_new", "ystr_len", "String_free"]);
    emitter.set_optimize_runtime(true);
    emitter.set_optimize_helper_effects(true);
    // `load` is reserved in source. Deliberately bypass parsing of its name
    // to inspect the emitter's proof; the invalid scalar-as-pointer load is
    // never compiled or executed.
    assert!(!emitter
        .emit_program(&ast, &y::cpu_jit::host_profile())
        .contains("runtime.read"));
}
