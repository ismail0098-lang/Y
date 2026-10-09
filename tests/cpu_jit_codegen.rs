#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::time::Duration;
use y::cpu_jit::{CpuJit, JitOptions, JitValue as V};

const SOURCE: &str = r#"
fn choose(x: I64) -> I64 {
    if x < 0 { return x - 1; }
    return x * 3 + 2;
}
fn fib(n: I64) -> I64 {
    if n <= 1 { return n; }
    return fib(n - 1) + fib(n - 2);
}
fn round(a: F64, b: F64, c: F64) -> F64 { return a * b + c; }
fn float_bits(x: F64) -> F64 { return x; }
@unsafe
fn memory(values: GlobalMemory<I64>, seed: I64) -> I64 {
    let mut total: I64 = 0;
    for i in 0..8 {
        values[i] = choose(values[i] + seed);
        total += values[i] * (i + 1);
    }
    return total;
}
"#;

fn check(jit: &CpuJit, instrumented: bool) {
    let flat = jit.compile_timings();
    let details = jit.materialization_timings();
    assert_eq!(flat.accounted_duration(), flat.total);
    assert_eq!(details.accounted_duration(), details.total);
    assert_eq!(details.total, flat.materialization);
    assert!(details.submission > Duration::ZERO);
    assert!(details.first_lookup > Duration::ZERO);
    assert_eq!(details.function_lookup_count, 10); // Five public functions/adapters.
    assert_eq!(details.profile_lookup_count, u64::from(instrumented));
    if !instrumented {
        assert_eq!(details.profile_lookup, Duration::ZERO);
    }
    assert_eq!(
        details.first_lookup_before_object.is_some(),
        details.first_lookup_after_object.is_some()
    );
    if let (Some(before), Some(after)) = (
        details.first_lookup_before_object,
        details.first_lookup_after_object,
    ) {
        assert!(details.object_observer_available);
        assert_eq!(details.object_count, 1);
        assert_eq!(before + after, details.first_lookup);
        assert!(before > Duration::ZERO);
        assert!(after > Duration::ZERO);
    }
    if details.object_observer_available {
        assert!(details.object_count > 0);
        assert!(details.object_bytes > 0);
    } else {
        assert_eq!(details.object_count, 0);
        assert_eq!(details.object_bytes, 0);
        assert!(details.first_lookup_before_object.is_none());
    }
    let snapshot = *details;
    for seed in [-11_i64, 0, 19] {
        let mut actual = [-41_i64, -3, 0, 1, 9, -19, 13, -21];
        let expected = actual.map(|value| {
            let x = value + seed;
            if x < 0 {
                x - 1
            } else {
                x * 3 + 2
            }
        });
        let checksum: i64 = expected
            .iter()
            .enumerate()
            .map(|(i, x)| x * (i as i64 + 1))
            .sum();
        assert_eq!(
            unsafe {
                jit.call(
                    "memory",
                    &[V::Pointer(actual.as_mut_ptr().cast()), V::I64(seed)],
                )
                .unwrap()
            },
            V::I64(checksum)
        );
        assert_eq!(actual, expected);
    }
    let native_fib: unsafe extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(jit.function_address("fib").unwrap()) };
    for n in [0_i64, 1, 2, 9, 15] {
        let (mut a, mut b) = (0, 1);
        for _ in 0..n {
            (a, b) = (b, a + b);
        }
        assert_eq!(unsafe { native_fib(n) }, a);
        assert_eq!(unsafe { jit.call("fib", &[V::I64(n)]).unwrap() }, V::I64(a));
    }
    // This product rounds to exactly one before addition; an FMA differs.
    assert_eq!(
        unsafe {
            jit.call(
                "round",
                &[
                    V::F64(f64::from_bits(0x3ff0000000000001)),
                    V::F64(f64::from_bits(0x3feffffffffffffe)),
                    V::F64(-1.0),
                ],
            )
            .unwrap()
        }
        .bits(),
        0
    );
    for bits in [
        0x8000000000000000_u64,
        0x7ff8000000001234,
        0x7ff0000000000000,
    ] {
        assert_eq!(
            unsafe {
                jit.call("float_bits", &[V::F64(f64::from_bits(bits))])
                    .unwrap()
            }
            .bits(),
            bits
        );
    }
    assert_eq!(*jit.materialization_timings(), snapshot);
}

#[test]
fn codegen_levels_preserve_ir_profiles_and_full_results() {
    assert_eq!(JitOptions::default().codegen_opt_level, None);
    for level in [0, 3] {
        let inherited = JitOptions {
            opt_level: level,
            ..JitOptions::default()
        };
        let original = CpuJit::compile_with_options(SOURCE, inherited).unwrap();
        let training = CpuJit::compile_instrumented(SOURCE, inherited).unwrap();
        check(&original, false);
        check(&training, true);
        let profile = training.branch_profile().unwrap();
        let profiled = CpuJit::compile_with_profile(SOURCE, inherited, &profile).unwrap();
        check(&profiled, false);
        for codegen in 0..=3 {
            let options = JitOptions {
                codegen_opt_level: Some(codegen),
                ..inherited
            };
            let ordinary = CpuJit::compile_with_options(SOURCE, options).unwrap();
            let instrumented = CpuJit::compile_instrumented(SOURCE, options).unwrap();
            assert_eq!(ordinary.optimized_ir(), original.optimized_ir());
            assert_eq!(instrumented.optimized_ir(), training.optimized_ir());
            check(&ordinary, false);
            check(&instrumented, true);
            assert_eq!(instrumented.branch_profile().unwrap(), profile);
            let optimized = CpuJit::compile_with_profile(SOURCE, options, &profile).unwrap();
            assert_eq!(optimized.optimized_ir(), profiled.optimized_ir());
            check(&optimized, false);
            assert_eq!(training.branch_profile().unwrap(), profile);
        }
    }
}

#[test]
fn invalid_codegen_level_is_refused_before_llvm() {
    let error = match CpuJit::compile_with_options(
        "fn value() -> I64 { return 3; }",
        JitOptions {
            codegen_opt_level: Some(4),
            ..JitOptions::default()
        },
    ) {
        Ok(_) => panic!("invalid codegen level was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("codegen_opt_level"), "{error}");
}
