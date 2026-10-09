#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use y::cpu_jit::{CpuJit, JitOptions, JitValue as V};

const ABI_SOURCE: &str = r#"
fn signed(x: I8) -> I8 { return x; }
fn unsigned(x: U64) -> U64 { return x; }
fn boolean(x: bool) -> bool { return x; }
fn single(x: F32) -> F32 { return x; }
fn double(x: F64) -> F64 { return x; }
@unsafe
fn pointer(x: GlobalMemory<I64>) -> GlobalMemory<I64> { return x; }
@unsafe
fn write(x: GlobalMemory<I64>, value: I64) { x[0] = value; }
fn helper(x: I64) -> I64 { return x * 7 + 3; }
fn outer(x: I64) -> I64 { return helper(x) + helper(x + 1); }
"#;

const LOOP_SOURCE: &str = r#"
@unsafe
fn work(n: I64, seed: F64) -> F64 {
    let mut x: F64 = seed;
    let mut i: I64 = 0;
    while i < n {
        x = x * 1.0000001 + 0.00000031;
        if x > 1.0 { x = x - 1.0; }
        i = i + 1;
    }
    return x;
}
"#;

fn options(opt_level: u8, optimize_call_adapters: bool) -> JitOptions {
    JitOptions {
        opt_level,
        optimize_call_adapters,
        ..JitOptions::default()
    }
}

fn definitions(ir: &str) -> Vec<&str> {
    ir.match_indices("define ")
        .map(|(start, _)| {
            let remaining = &ir[start..];
            &remaining[..remaining.find("\n}").unwrap() + 2]
        })
        .collect()
}

fn definition<'a>(ir: &'a str, name: &str) -> &'a str {
    definitions(ir)
        .into_iter()
        .find(|body| body.lines().next().unwrap().contains(&format!("@{name}(")))
        .unwrap_or_else(|| panic!("missing definition {name}"))
}

fn adapter_for<'a>(ir: &'a str, source_name: &str) -> &'a str {
    definitions(ir)
        .into_iter()
        .find(|body| {
            body.lines().next().unwrap().contains("@__y_jit_dispatch_")
                && body.lines().any(|line| {
                    line.contains("call ") && line.contains(&format!("@{source_name}("))
                })
        })
        .unwrap_or_else(|| panic!("missing adapter call to {source_name}"))
}

fn has_noinline(ir: &str, line: &str) -> bool {
    line.split_whitespace().any(|token| {
        if token == "noinline" {
            return true;
        }
        let token = token.trim_end_matches(',');
        if !token.starts_with('#') {
            return false;
        }
        ir.lines()
            .find(|attributes| attributes.starts_with(&format!("attributes {token} =")))
            .is_some_and(|attributes| attributes.split_whitespace().any(|attr| attr == "noinline"))
    })
}

#[test]
fn compact_and_original_adapters_preserve_scalar_float_pointer_and_void_calls() {
    for opt_level in [0, 3] {
        for compact in [false, true] {
            let jit =
                CpuJit::compile_with_options(ABI_SOURCE, options(opt_level, compact)).unwrap();
            for (name, argument) in [
                ("signed", V::I8(i8::MIN)),
                ("unsigned", V::U64(u64::MAX)),
                ("boolean", V::Bool(false)),
                ("boolean", V::Bool(true)),
                ("single", V::F32(-0.0)),
                ("single", V::F32(f32::from_bits(0x7fc01234))),
                ("double", V::F64(-0.0)),
                ("double", V::F64(f64::from_bits(0x7ff8000000001234))),
            ] {
                let result = unsafe { jit.call(name, &[argument]) }.unwrap();
                assert_eq!(result.abi_type(), argument.abi_type(), "{name}");
                assert_eq!(result.bits(), argument.bits(), "{name}");
            }
            let mut memory = [71_i64, 72];
            let pointer = V::Pointer(memory.as_mut_ptr().cast());
            assert_eq!(unsafe { jit.call("pointer", &[pointer]) }.unwrap(), pointer);
            assert!(unsafe { jit.call("write", &[pointer, V::U64(19)]) }.is_err());
            assert_eq!(memory, [71, 72]);
            assert_eq!(
                unsafe { jit.call("write", &[pointer, V::I64(4294967313)]) }.unwrap(),
                V::Void
            );
            assert_eq!(memory, [4294967313, 72]);
            assert_eq!(
                unsafe { jit.call("outer", &[V::I64(5)]) }.unwrap(),
                V::I64(83)
            );
            assert!(!jit
                .functions()
                .any(|name| name.starts_with("__y_jit_dispatch_")));
        }
    }
}

#[test]
fn noinline_is_only_attached_to_adapter_call_sites() {
    for compact in [false, true] {
        let jit = CpuJit::compile_with_options(ABI_SOURCE, options(0, compact)).unwrap();
        let ir = jit.optimized_ir();
        for name in [
            "signed", "unsigned", "boolean", "single", "double", "pointer", "write", "helper",
            "outer",
        ] {
            let source = definition(ir, name);
            assert!(!has_noinline(ir, source.lines().next().unwrap()), "{name}");
            for call in source.lines().filter(|line| line.contains("call ")) {
                assert!(!has_noinline(ir, call), "source call: {call}");
            }
            let adapter = adapter_for(ir, name);
            assert!(!has_noinline(ir, adapter.lines().next().unwrap()));
            let call = adapter.lines().find(|line| line.contains("call ")).unwrap();
            assert_eq!(has_noinline(ir, call), compact, "adapter call: {call}");
        }
    }
    let jit = CpuJit::compile_with_options(ABI_SOURCE, options(3, true)).unwrap();
    assert!(!definition(jit.optimized_ir(), "outer").contains("@helper("));
}

#[test]
fn optimized_loop_adapter_calls_one_native_body_without_copying_control_flow() {
    for compact in [false, true] {
        let jit = CpuJit::compile_with_options(LOOP_SOURCE, options(3, compact)).unwrap();
        let ir = jit.optimized_ir();
        assert!(definition(ir, "work")
            .lines()
            .any(|line| line.trim_start().starts_with("br ")));
        if compact {
            let adapter = adapter_for(ir, "work");
            assert!(!adapter.lines().any(|line| {
                let instruction = line.trim_start();
                instruction.starts_with("br ")
                    || instruction.contains(" = phi ")
                    || instruction.contains(" fmul ")
                    || instruction.contains(" fadd ")
            }));
            assert_eq!(
                adapter
                    .lines()
                    .filter(|line| line.contains("call "))
                    .count(),
                1
            );
        }
        let native: unsafe extern "C" fn(i64, f64) -> f64 =
            unsafe { std::mem::transmute(jit.function_address("work").unwrap()) };
        for n in [0_i64, 1, 37, 10000] {
            let mut expected = 0.875_f64;
            for _ in 0..n {
                expected = expected * 1.0000001 + 0.00000031;
                if expected > 1.0 {
                    expected -= 1.0;
                }
            }
            let checked = unsafe { jit.call("work", &[V::I64(n), V::F64(0.875)]) }.unwrap();
            assert_eq!(checked.bits(), expected.to_bits());
            assert_eq!(unsafe { native(n, 0.875) }.to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn unavailable_runtime_still_fails_during_compilation_before_any_lookup_or_call() {
    for opt_level in [0, 3] {
        for compact in [false, true] {
            let result = CpuJit::compile_with_options(
                "fn main() -> I32 { return init_shadowplay_gui(); }",
                options(opt_level, compact),
            );
            let error = match result {
                Ok(_) => panic!("missing runtime compiled successfully"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains("init_shadowplay_gui"), "{error}");
            assert!(error.contains("unavailable"), "{error}");
        }
    }
}
