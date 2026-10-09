#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::time::Duration;
use y::cpu_jit::{CpuJit, JitOptions, JitValue as V};

const SOURCE: &str = r#"
fn choose(x: I64) -> I64 {
    if x < 0 { return x - 1; }
    return x * 3 + 2;
}
fn weight(x: I64) -> I64 { return (x & 7) + 1; }
@unsafe
fn memory(values: &mut [I64; 8], seed: I64) -> I64 {
    let mut result: I64 = 0;
    for i in 0..8 {
        values[i] = choose(values[i] + seed);
        result += values[i] * weight(i);
    }
    return result;
}
@unsafe
fn owned(n: I64, release: bool) -> I64 {
    let mut text: String = "ab";
    let mut i: I64 = 0;
    while i < n { String_push(&mut text, 'z'); i += 1; }
    let value: I64 = choose(n);
    if release { String_free(&mut text); }
    let result: I64 = value * 100 + String_len(&text);
    String_free(&mut text);
    return result;
}
"#;

fn check(jit: &CpuJit, level: u8, profiled: bool) {
    let timings = jit.compile_timings();
    let details = jit.optimization_timings();
    assert_eq!(timings.accounted_duration(), timings.total);
    assert_eq!(details.accounted_duration(), details.total);
    assert_eq!(details.total, timings.optimization);
    assert!(details.pipeline > Duration::ZERO);
    assert!(timings.verification > Duration::ZERO);
    let selected = profiled && level != 0 && jit.profiled_branches() != 0;
    assert_eq!(timings.verification_checks, if selected { 3 } else { 2 });
    if !selected {
        assert_eq!(details.profile_selection, Duration::ZERO);
    }
    let before = *timings;
    let details_before = *details;
    for seed in [-7_i64, 0, 13] {
        let mut actual = [-91_i64, 0, 7, -2, 13, 4, -15, 8];
        let expected = actual.map(|x| {
            let value = x + seed;
            if value < 0 {
                value - 1
            } else {
                value * 3 + 2
            }
        });
        let checksum: i64 = expected
            .iter()
            .enumerate()
            .map(|(i, value)| value * (i as i64 + 1))
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
    for n in [-3_i64, 0, 9] {
        for release in [false, true] {
            let scalar = if n < 0 { n - 1 } else { n * 3 + 2 };
            let length = if release { 0 } else { 2 + n.max(0) };
            assert_eq!(
                unsafe { jit.call("owned", &[V::I64(n), V::Bool(release)]).unwrap() },
                V::I64(scalar * 100 + length)
            );
        }
    }
    assert_eq!(*jit.compile_timings(), before);
    assert_eq!(*jit.optimization_timings(), details_before);
}

#[test]
fn both_verification_policies_preserve_profiles_ir_and_full_results() {
    assert!(JitOptions::default().verify_each_pass);
    for level in [0, 3] {
        let strict = JitOptions {
            opt_level: level,
            verify_each_pass: true,
            ..JitOptions::default()
        };
        let boundary = JitOptions {
            verify_each_pass: false,
            ..strict
        };
        let ordinary_strict = CpuJit::compile_with_options(SOURCE, strict).unwrap();
        let ordinary_boundary = CpuJit::compile_with_options(SOURCE, boundary).unwrap();
        assert_eq!(
            ordinary_strict.optimized_ir(),
            ordinary_boundary.optimized_ir()
        );
        check(&ordinary_strict, level, false);
        check(&ordinary_boundary, level, false);
        let training_strict = CpuJit::compile_instrumented(SOURCE, strict).unwrap();
        let training_boundary = CpuJit::compile_instrumented(SOURCE, boundary).unwrap();
        assert_eq!(
            training_strict.optimized_ir(),
            training_boundary.optimized_ir()
        );
        check(&training_strict, level, false);
        check(&training_boundary, level, false);
        let profile = training_strict.branch_profile().unwrap();
        assert_eq!(profile, training_boundary.branch_profile().unwrap());
        let optimized_strict = CpuJit::compile_with_profile(SOURCE, strict, &profile).unwrap();
        // Policy changes verification, not original-IR profile identity.
        let optimized_boundary = CpuJit::compile_with_profile(SOURCE, boundary, &profile).unwrap();
        assert_eq!(
            optimized_strict.optimized_ir(),
            optimized_boundary.optimized_ir()
        );
        check(&optimized_strict, level, true);
        check(&optimized_boundary, level, true);
        assert_eq!(profile, training_strict.branch_profile().unwrap());
    }
}
