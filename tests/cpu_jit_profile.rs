#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use y::cpu_jit::{CpuJit, CpuJitCache, JitOptions, JitValue};

const SOURCE: &str = r#"
fn choose(x: I64) -> I64 {
    if x < 0 { return x - 1; }
    return x + 2;
}
fn recurse(x: I64) -> I64 {
    if x <= 1 { return 1; }
    return x * recurse(x - 1);
}
fn logical(x: I64, y: I64) -> bool { return x > 0 && y > 0; }
@unsafe
fn recurrence(n: I64, seed: F64) -> F64 {
    let mut x: F64 = seed;
    let mut i: I64 = 0;
    while i < n {
        x = x * 1.000001 + 0.0000003;
        if x > 2.0 { x = x - 1.9999; }
        i = i + 1;
    }
    return x;
}
fn rotate(x: U64) -> U64 { return (x << 13) | (x >> 51); }
"#;

fn options(opt_level: u8) -> JitOptions {
    JitOptions {
        opt_level,
        ..JitOptions::default()
    }
}

#[test]
fn counters_record_actual_taken_edges_and_snapshots_are_independent() {
    for opt_level in [0, 3] {
        let training = CpuJit::compile_instrumented(SOURCE, options(opt_level)).unwrap();
        let empty = training.branch_profile().unwrap();
        assert_eq!(
            empty.total_observations(),
            0,
            "compile must not execute source"
        );
        assert!(!empty.sites().is_empty());
        let choose: unsafe extern "C" fn(i64) -> i64 =
            unsafe { std::mem::transmute(training.function_address("choose").unwrap()) };
        for x in [-3, -1, 0, 2, 5] {
            assert_eq!(unsafe { choose(x) }, if x < 0 { x - 1 } else { x + 2 });
        }
        let measured = training.branch_profile().unwrap();
        let site = measured
            .sites()
            .iter()
            .find(|site| site.function == "choose")
            .unwrap();
        assert_eq!((site.true_count, site.false_count), (2, 3));
        assert_eq!(empty.total_observations(), 0);
        assert_eq!(
            unsafe { training.call("choose", &[JitValue::I64(-9)]).unwrap() },
            JitValue::I64(-10)
        );
        let newer = training.branch_profile().unwrap();
        let site = newer
            .sites()
            .iter()
            .find(|site| site.function == "choose")
            .unwrap();
        assert_eq!((site.true_count, site.false_count), (3, 3));
        assert_eq!(measured.total_observations(), 5);
        // Profile globals are an implementation detail, not function entrypoints.
        assert!(!training
            .functions()
            .any(|name| name.contains("branch_counters")));
    }
}

#[test]
fn measured_profiles_recompile_without_counters_and_preserve_unseen_inputs() {
    for opt_level in [0, 3] {
        let training = CpuJit::compile_instrumented(SOURCE, options(opt_level)).unwrap();
        for _ in 0..1000 {
            assert_eq!(
                unsafe { training.call("choose", &[JitValue::I64(8)]).unwrap() },
                JitValue::I64(10)
            );
        }
        let profile = training.branch_profile().unwrap();
        let optimized = CpuJit::compile_with_profile(SOURCE, options(opt_level), &profile).unwrap();
        assert_eq!(optimized.profiled_branches(), 1);
        assert!(!optimized.optimized_ir().contains("atomicrmw"));
        assert!(optimized
            .branch_profile()
            .unwrap_err()
            .to_string()
            .contains("instrumentation"));
        // An unobserved edge remains semantically possible.
        for x in [-50, -1, 0, 17] {
            assert_eq!(
                unsafe { optimized.call("choose", &[JitValue::I64(x)]).unwrap() },
                JitValue::I64(if x < 0 { x - 1 } else { x + 2 })
            );
        }
        assert_eq!(
            unsafe { optimized.call("recurse", &[JitValue::I64(8)]).unwrap() },
            JitValue::I64(40320)
        );
        for (x, y) in [(0, 8), (1, 8), (1, 0), (-1, -1)] {
            assert_eq!(
                unsafe {
                    optimized
                        .call("logical", &[JitValue::I64(x), JitValue::I64(y)])
                        .unwrap()
                },
                JitValue::Bool(x > 0 && y > 0)
            );
        }
        // Recompilation does not invalidate old executable addresses or counters.
        assert_eq!(
            unsafe { training.call("choose", &[JitValue::I64(-1)]).unwrap() },
            JitValue::I64(-2)
        );
        drop(training);
        assert_eq!(
            unsafe { optimized.call("choose", &[JitValue::I64(3)]).unwrap() },
            JitValue::I64(5)
        );
    }
}

#[test]
fn stale_profiles_and_different_lowering_options_are_rejected() {
    let training = CpuJit::compile_instrumented(SOURCE, options(3)).unwrap();
    unsafe {
        training.call("choose", &[JitValue::I64(1)]).unwrap();
    }
    let profile = training.branch_profile().unwrap();
    let changed = SOURCE.replace("return x + 2", "return x + 7");
    assert!(CpuJit::compile_with_profile(&changed, options(3), &profile)
        .err()
        .unwrap()
        .to_string()
        .contains("profile"));
    let baseline = JitOptions {
        recognize_rotates: false,
        ..options(3)
    };
    assert!(CpuJit::compile_with_profile(SOURCE, baseline, &profile)
        .err()
        .unwrap()
        .to_string()
        .contains("profile"));
    assert!(CpuJit::compile_with_options(SOURCE, options(3))
        .unwrap()
        .branch_profile()
        .is_err());
}

#[test]
fn profile_guided_recurrence_keeps_ieee_results_and_conditional_work() {
    let training = CpuJit::compile_instrumented(SOURCE, options(3)).unwrap();
    let train: unsafe extern "C" fn(i64, f64) -> f64 =
        unsafe { std::mem::transmute(training.function_address("recurrence").unwrap()) };
    unsafe {
        train(1_000_000, 0.5);
    }
    let profile = training.branch_profile().unwrap();
    let optimized = CpuJit::compile_with_profile(SOURCE, options(3), &profile).unwrap();
    let baseline = CpuJit::compile_with_options(SOURCE, options(3)).unwrap();
    let optimized_fn: unsafe extern "C" fn(i64, f64) -> f64 =
        unsafe { std::mem::transmute(optimized.function_address("recurrence").unwrap()) };
    let baseline_fn: unsafe extern "C" fn(i64, f64) -> f64 =
        unsafe { std::mem::transmute(baseline.function_address("recurrence").unwrap()) };
    for (n, seed) in [
        (0, -0.0),
        (-4, 0.5),
        (1000, -2.0),
        (1000, 0.5),
        (1000, 8.0),
        (1_000_000, 0.5),
        (100, f64::INFINITY),
    ] {
        assert_eq!(
            unsafe { optimized_fn(n, seed) }.to_bits(),
            unsafe { baseline_fn(n, seed) }.to_bits()
        );
    }
    assert!(unsafe { optimized_fn(100, f64::NAN) }.is_nan());
    assert_eq!(optimized.profiled_branches(), 1);
    let function = optimized
        .optimized_ir()
        .split("define double @recurrence(")
        .nth(1)
        .unwrap()
        .split("\n}")
        .next()
        .unwrap();
    assert!(
        !function.contains("select i1"),
        "measured rare work should be conditional:\n{function}"
    );
    assert!(!function.contains("fmul fast") && !function.contains("fadd fast"));
}

#[test]
fn branchless_programs_and_cache_rotation_settings_are_handled() {
    let source = "fn answer() -> I32 { return 42; }";
    let training = CpuJit::compile_instrumented(source, options(3)).unwrap();
    let profile = training.branch_profile().unwrap();
    assert_eq!(profile.total_observations(), 0);
    assert!(profile.sites().is_empty());
    let optimized = CpuJit::compile_with_profile(source, options(3), &profile).unwrap();
    assert_eq!(optimized.profiled_branches(), 0);
    assert_eq!(
        unsafe { optimized.call("answer", &[]).unwrap() },
        JitValue::I32(42)
    );
    let mut cache = CpuJitCache::new(2);
    let rotate = cache.compile(SOURCE, options(3)).unwrap();
    let baseline = cache
        .compile(
            SOURCE,
            JitOptions {
                recognize_rotates: false,
                ..options(3)
            },
        )
        .unwrap();
    assert!(!std::rc::Rc::ptr_eq(&rotate, &baseline));
    assert_eq!(cache.misses(), 2);
}

#[test]
fn loop_control_policy_keeps_counts_and_profiles_nested_work_and_breaks() {
    let source = r#"
    @unsafe
    fn count(n: I64) -> I64 {
        let mut i: I64 = 0;
        while i < n { i = i + 1; }
        return i;
    }
    @unsafe
    fn nested(n: I64, stop: I64) -> I64 {
        let mut sum: I64 = 0;
        let mut i: I64 = 0;
        while i < n {
            let mut j: I64 = 0;
            while j < 5 {
                if j == stop { break; }
                if ((i + j) & 1) == 0 { sum = sum + i + j; }
                j = j + 1;
            }
            i = i + 1;
        }
        return sum;
    }
    "#;
    for opt_level in [0, 3] {
        let training = CpuJit::compile_instrumented(source, options(opt_level)).unwrap();
        unsafe {
            assert_eq!(
                training.call("count", &[JitValue::I64(5)]).unwrap(),
                JitValue::I64(5)
            );
            training
                .call("nested", &[JitValue::I64(4), JitValue::I64(3)])
                .unwrap();
        }
        let profile = training.branch_profile().unwrap();
        assert_eq!(
            profile.sites().len(),
            5,
            "all original branches are still captured"
        );
        let count = profile
            .sites()
            .iter()
            .find(|site| site.function == "count")
            .unwrap();
        assert_eq!((count.true_count, count.false_count), (5, 1));
        let conservative =
            CpuJit::compile_with_profile(source, options(opt_level), &profile).unwrap();
        let all_edges = CpuJit::compile_with_profile(
            source,
            JitOptions {
                profile_loop_controls: true,
                ..options(opt_level)
            },
            &profile,
        )
        .unwrap();
        assert_eq!(
            conservative.profiled_branches(),
            2,
            "inner work and breaks stay weighted"
        );
        assert_eq!(all_edges.profiled_branches(), 5);
        if opt_level == 0 {
            let function = conservative
                .optimized_ir()
                .split("define i64 @count(")
                .nth(1)
                .unwrap()
                .split("\n}")
                .next()
                .unwrap();
            assert!(
                !function.contains("!prof"),
                "loop-only code retains default policy"
            );
            let function = conservative
                .optimized_ir()
                .split("define i64 @nested(")
                .nth(1)
                .unwrap()
                .split("\n}")
                .next()
                .unwrap();
            assert_eq!(function.matches("!prof").count(), 2);
        }
        for (n, stop) in [
            (0, 3),
            (-4, 0),
            (6, -1),
            (6, 0),
            (6, 2),
            (6, 3),
            (6, 5),
            (6, 100),
        ] {
            let mut expected = 0;
            for i in 0..n {
                for j in 0..5 {
                    if j == stop {
                        break;
                    }
                    if (i + j) & 1 == 0 {
                        expected += i + j;
                    }
                }
            }
            for jit in [&conservative, &all_edges] {
                assert_eq!(
                    unsafe {
                        jit.call("nested", &[JitValue::I64(n), JitValue::I64(stop)])
                            .unwrap()
                    },
                    JitValue::I64(expected),
                    "O{opt_level}: n={n}, stop={stop}"
                );
            }
        }
        assert_eq!(
            profile
                .sites()
                .iter()
                .find(|site| site.function == "count")
                .unwrap()
                .true_count,
            5
        );
    }
}

#[test]
fn concurrent_native_calls_record_each_observation_atomically() {
    use std::sync::{Arc, Barrier};
    for training_opt_level in [None, Some(1)] {
        let training = CpuJit::compile_instrumented(
            SOURCE,
            JitOptions {
                training_opt_level,
                ..options(3)
            },
        )
        .unwrap();
        let address = training.function_address("choose").unwrap();
        let barrier = Arc::new(Barrier::new(5));
        let threads: Vec<_> = (0..4)
            .map(|thread| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    // This scalar function shares only the compiler's atomic counters.
                    // The creating thread retains the JIT until every worker joins.
                    let choose: unsafe extern "C" fn(i64) -> i64 =
                        unsafe { std::mem::transmute(address) };
                    barrier.wait();
                    for i in 0..4096 {
                        let x = if (thread + i) % 2 == 0 { -1 } else { 0 };
                        assert_eq!(unsafe { choose(x) }, if x < 0 { -2 } else { 2 });
                    }
                })
            })
            .collect();
        barrier.wait();
        let during = training.branch_profile().unwrap();
        assert!(during.total_observations() <= 4 * 4096);
        for thread in threads {
            thread.join().unwrap();
        }
        let final_profile = training.branch_profile().unwrap();
        let choose = final_profile
            .sites()
            .iter()
            .find(|site| site.function == "choose")
            .unwrap();
        assert_eq!((choose.true_count, choose.false_count), (8192, 8192));
        assert_eq!(final_profile.total_observations(), 16384);
    }
}

#[test]
fn unobserved_short_circuit_memory_paths_remain_guarded_and_callable() {
    let source = r#"
    @unsafe
    fn guarded(values: &mut [I64; 4], index: I64) -> I64 {
        if index >= 0 && index < 4 {
            values[index] = values[index] + 1;
            return values[index];
        }
        return -1;
    }
    "#;
    for opt_level in [0, 3] {
        let training = CpuJit::compile_instrumented(source, options(opt_level)).unwrap();
        let train: unsafe extern "C" fn(*mut i64, i64) -> i64 =
            unsafe { std::mem::transmute(training.function_address("guarded").unwrap()) };
        let mut values = [1, 2, 3, 4];
        for _ in 0..100 {
            assert_eq!(unsafe { train(values.as_mut_ptr(), -1) }, -1);
        }
        assert_eq!(values, [1, 2, 3, 4]);
        let optimized = CpuJit::compile_with_profile(
            source,
            options(opt_level),
            &training.branch_profile().unwrap(),
        )
        .unwrap();
        let guarded: unsafe extern "C" fn(*mut i64, i64) -> i64 =
            unsafe { std::mem::transmute(optimized.function_address("guarded").unwrap()) };
        for index in [-99, -1, 4, 99] {
            assert_eq!(unsafe { guarded(values.as_mut_ptr(), index) }, -1);
            assert_eq!(values, [1, 2, 3, 4]);
        }
        assert_eq!(unsafe { guarded(values.as_mut_ptr(), 2) }, 4);
        assert_eq!(values, [1, 2, 4, 4]);
    }
}
