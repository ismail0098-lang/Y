//! Edge-local atomic instrumentation must preserve exact original-site counts.
//! Independent bounded oracles inspect full values, overlapping buffers,
//! runtime object contents and IEEE bits, while select instrumentation provides
//! a separate exact-profile reference. These are correctness gates, not a
//! performance benchmark or a proof for arbitrary programs.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::time::Duration;
use y::cpu_jit::{BranchProfile, CpuJit, JitOptions, JitValue as V};

const SOURCE: &str = r#"
fn choose(x: I64) -> I64 {
    if x < 0 { return x - 1; }
    return x * 3 + 2;
}
fn fib(n: I64) -> I64 {
    if n <= 1 { return n; }
    return fib(n - 1) + fib(n - 2);
}
fn word_value(payload: I64, index: I64) -> I64 { return payload ^ index; }
@unsafe
fn memory(p: GlobalMemory<I64>, q: GlobalMemory<I64>, seed: I64, n: I64) -> I64 {
    let mut i: I64 = 0;
    while i < n {
        let old: I64 = p[i];
        p[i] = choose(q[7 - i] + seed);
        q[7 - i] = old;
        i = i + 1;
    }
    return p[0] ^ q[7];
}
@unsafe
fn owned(output: GlobalMemory<I64>, payload: I64, n: I64, token: char) {
    let mut text: String = "";
    let mut suffix: String = "xy";
    String_push(&mut suffix, '\0');
    let mut words: Vec = Vec_new(8);
    let mut i: I64 = 0;
    while i < n {
        let value: I64 = word_value(payload, i);
        Vec_push(&mut words, &value);
        if (i & 1) == 0 { String_push(&mut text, token); }
        else { String_push(&mut text, 'Z'); }
        i = i + 1;
    }
    String_push_str(&mut text, &suffix);
    String_push_str(&mut text, &text);
    i = 0;
    while i < Vec_len(&words) {
        let value: I64 = load(yvec_get(&words, i));
        output[i] = value;
        i = i + 1;
    }
    i = 0;
    while i < String_len(&text) {
        output[64 + i] = ychar_to_ascii(String_char_at(&text, i));
        i = i + 1;
    }
    output[160] = Vec_len(&words);
    output[161] = String_len(&text);
    output[162] = ychar_to_ascii(String_char_at(&text, -1));
    output[163] = ychar_to_ascii(String_char_at(&text, String_len(&text)));
    String_free(&mut text);
    String_free(&mut suffix);
    Vec_free(&mut words);
    output[164] = String_len(&text) + String_len(&suffix) + Vec_len(&words);
    output[165] = ychar_to_ascii(String_char_at(&text, 0));
    output[166] = ychar_to_ascii(Vec_get_char(&words, 0));
    String_free(&mut text);
    Vec_free(&mut words);
}
fn rounded64(a: F64, b: F64) -> F64 { let product: F64 = a * b; return product - 1.0; }
fn rounded32(a: F32, b: F32) -> F32 { let product: F32 = a * b; return product - 1.0; }
fn ieee64(x: F64) -> F64 { return x; }
fn ieee32(x: F32) -> F32 { return x; }
@unsafe
fn record(trace: GlobalMemory<I64>, value: I64) -> I64 {
    trace[0] = trace[0] + 1;
    return value;
}
@unsafe
fn short_circuit(trace: GlobalMemory<I64>, x: I64, y: I64) -> I64 {
    let mut result: I64 = 0;
    if x > 0 && record(trace, y) > 0 { result = result + 1; }
    if x < 0 || record(trace, y) < 0 { result = result + 2; }
    return result;
}
fn comparisons(a: F64, b: F64) -> I32 {
    let mut result: I32 = 0;
    if a == b { result += 1; }
    if a != b { result += 2; }
    if a < b { result += 4; }
    if a >= b { result += 8; }
    return result;
}
"#;

#[derive(Clone, Copy)]
struct Case {
    seed: i64,
    length: i64,
    fib: i64,
    token: u8,
}

const TRAINING: [Case; 3] = [
    Case {
        seed: -7,
        length: 0,
        fib: 0,
        token: 0,
    },
    Case {
        seed: 0,
        length: 3,
        fib: 2,
        token: 127,
    },
    Case {
        seed: 13,
        length: 6,
        fib: 5,
        token: 200,
    },
];
const UNSEEN: [Case; 3] = [
    Case {
        seed: -19,
        length: -1,
        fib: 1,
        token: 255,
    },
    Case {
        seed: 29,
        length: 17,
        fib: 9,
        token: 0,
    },
    Case {
        seed: -3,
        length: 31,
        fib: 13,
        token: 200,
    },
];

#[derive(Default)]
struct CounterOracle {
    choose_true: u64,
    choose_false: u64,
    fib_true: u64,
    fib_false: u64,
}

impl CounterOracle {
    fn choose(&mut self, x: i64) -> i64 {
        if x < 0 {
            self.choose_true += 1;
            x - 1
        } else {
            self.choose_false += 1;
            x * 3 + 2
        }
    }

    fn fib(&mut self, n: i64) -> i64 {
        // A recursive Fibonacci tree has F(n+1) leaves and one fewer
        // non-leaf calls. Compute both via an iterative sequence.
        let (mut older, mut newer) = (0_i64, 1_i64);
        for _ in 0..n {
            (older, newer) = (newer, older + newer);
        }
        self.fib_true += newer as u64;
        self.fib_false += newer as u64 - 1;
        older
    }

    fn check(&self, profile: &BranchProfile, context: &str) {
        for (name, yes, no) in [
            ("choose", self.choose_true, self.choose_false),
            ("fib", self.fib_true, self.fib_false),
        ] {
            let sites: Vec<_> = profile
                .sites()
                .iter()
                .filter(|site| site.function == name)
                .collect();
            assert_eq!(sites.len(), 1, "{context}: {name} original branch");
            assert_eq!(
                (sites[0].true_count, sites[0].false_count),
                (yes, no),
                "{context}: independently counted {name} outcomes"
            );
        }
    }
}

fn call(jit: &CpuJit, name: &str, values: &[V], context: &str) -> V {
    // Pointer values below address live complete host buffers on this thread;
    // scalar values carry the exact source-derived checked ABI.
    unsafe { jit.call(name, values) }
        .unwrap_or_else(|error| panic!("{context}: {name}: {error}\n{SOURCE}"))
}

fn check_case(jit: &CpuJit, case: Case, oracle: &mut CounterOracle, context: &str) {
    assert_eq!(
        call(jit, "choose", &[V::I64(case.seed)], context),
        V::I64(oracle.choose(case.seed)),
        "{context}: scalar branch"
    );
    let fibonacci = oracle.fib(case.fib);
    assert_eq!(
        call(jit, "fib", &[V::I64(case.fib)], context),
        V::I64(fibonacci)
    );
    let native_fib: unsafe extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(jit.function_address("fib").unwrap()) };
    assert_eq!(
        unsafe { native_fib(case.fib) },
        oracle.fib(case.fib),
        "{context}: direct recursive native entrypoint"
    );

    // Inspect all words and outer canaries, with both overlapping and
    // disjoint input/output regions. The oracle performs the specified
    // sequential state transitions, independently of LLVM execution.
    for second_base in [4_usize, 9] {
        let mut actual = std::array::from_fn::<_, 18, _>(|i| (i as i64 - 9) * 7);
        let mut expected = actual;
        for i in 0..8 {
            let old = expected[1 + i];
            let incoming = expected[second_base + 7 - i] + case.seed;
            expected[1 + i] = oracle.choose(incoming);
            expected[second_base + 7 - i] = old;
        }
        let returned = call(
            jit,
            "memory",
            &[
                V::Pointer(unsafe { actual.as_mut_ptr().add(1) }.cast()),
                V::Pointer(unsafe { actual.as_mut_ptr().add(second_base) }.cast()),
                V::I64(case.seed),
                V::I64(8),
            ],
            context,
        );
        assert_eq!(
            actual, expected,
            "{context}: complete aliasing buffer base={second_base}"
        );
        assert_eq!(returned, V::I64(expected[1] ^ expected[second_base + 7]));
    }

    const CANARY: i64 = 0x1234_5678_9abc_def;
    let mut actual = [CANARY; 194];
    let mut expected = actual;
    let n = case.length.max(0) as usize;
    assert!(n <= 31);
    let mut bytes: Vec<u8> = (0..n)
        .map(|i| if i % 2 == 0 { case.token } else { b'Z' })
        .collect();
    bytes.extend_from_slice(b"xy\0");
    let repeated = bytes.clone();
    bytes.extend_from_slice(&repeated);
    for i in 0..n {
        expected[1 + i] = case.seed ^ i as i64;
    }
    for (i, &byte) in bytes.iter().enumerate() {
        expected[1 + 64 + i] = i64::from(byte);
    }
    expected[1 + 160] = n as i64;
    expected[1 + 161] = bytes.len() as i64;
    expected[1 + 162..=1 + 166].fill(0);
    assert_eq!(
        call(
            jit,
            "owned",
            &[
                V::Pointer(unsafe { actual.as_mut_ptr().add(1) }.cast()),
                V::I64(case.seed),
                V::I64(case.length),
                V::U8(case.token),
            ],
            context
        ),
        V::Void
    );
    assert_eq!(
        actual, expected,
        "{context}: complete String/Vec contents, freed slots and canaries"
    );
}

fn check_ieee(jit: &CpuJit, context: &str) {
    for bits in [
        0_u64,
        0x8000_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_1234,
    ] {
        assert_eq!(
            call(jit, "ieee64", &[V::F64(f64::from_bits(bits))], context).bits(),
            bits,
            "{context}: F64 payload bits"
        );
    }
    for bits in [0_u32, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_1234] {
        assert_eq!(
            call(jit, "ieee32", &[V::F32(f32::from_bits(bits))], context).bits(),
            u64::from(bits),
            "{context}: F32 payload bits"
        );
    }
    // Each product rounds to exactly 1 before subtraction. Contracting the
    // two operations into an FMA would leave a nonzero negative result.
    assert_eq!(
        call(
            jit,
            "rounded64",
            &[
                V::F64(f64::from_bits(0x3ff0_0000_0000_0001)),
                V::F64(f64::from_bits(0x3fef_ffff_ffff_fffe))
            ],
            context
        )
        .bits(),
        0
    );
    assert_eq!(
        call(
            jit,
            "rounded32",
            &[
                V::F32(f32::from_bits(0x3f80_0001)),
                V::F32(f32::from_bits(0x3f7f_fffe))
            ],
            context
        )
        .bits(),
        0
    );
    for (a, b) in [
        (f64::NAN, 0.0),
        (0.0, f64::NAN),
        (-0.0, 0.0),
        (f64::NEG_INFINITY, f64::INFINITY),
        (1.25, -2.5),
    ] {
        let expected = i32::from(a == b)
            + 2 * i32::from(a != b)
            + 4 * i32::from(a < b)
            + 8 * i32::from(a >= b);
        assert_eq!(
            call(jit, "comparisons", &[V::F64(a), V::F64(b)], context),
            V::I32(expected)
        );
    }
}

fn exercise(jit: &CpuJit, cases: &[Case], context: &str) -> CounterOracle {
    let timings = *jit.compile_timings();
    let optimization = *jit.optimization_timings();
    let materialization = *jit.materialization_timings();
    assert_eq!(timings.accounted_duration(), timings.total);
    assert_eq!(optimization.accounted_duration(), optimization.total);
    assert_eq!(optimization.total, timings.optimization);
    assert_eq!(materialization.accounted_duration(), materialization.total);
    assert_eq!(materialization.total, timings.materialization);
    assert!(optimization.pipeline > Duration::ZERO);
    assert!(timings.verification > Duration::ZERO);
    let mut oracle = CounterOracle::default();
    for &case in cases {
        check_case(jit, case, &mut oracle, context);
    }
    check_ieee(jit, context);
    // AND/OR lowering has a PHI merge. Instrumentation must retain guarded
    // evaluation and all source-side effects when edge splitting falls back.
    for x in [-1_i64, 0, 1] {
        for y in [-1_i64, 0, 1] {
            let mut actual = [91_i64, 0, -73];
            let expected_count = i64::from(x > 0) + i64::from(x >= 0);
            let expected_result = i64::from(x > 0 && y > 0) + 2 * i64::from(x < 0 || y < 0);
            assert_eq!(
                call(
                    jit,
                    "short_circuit",
                    &[
                        V::Pointer(unsafe { actual.as_mut_ptr().add(1) }.cast()),
                        V::I64(x),
                        V::I64(y),
                    ],
                    context,
                ),
                V::I64(expected_result),
                "{context}: short-circuit x={x}, y={y}"
            );
            assert_eq!(
                actual,
                [91, expected_count, -73],
                "{context}: complete short-circuit trace and canaries"
            );
        }
    }
    assert_eq!(
        *jit.compile_timings(),
        timings,
        "{context}: execution excluded from compile timings"
    );
    assert_eq!(*jit.optimization_timings(), optimization);
    assert_eq!(*jit.materialization_timings(), materialization);
    oracle
}

#[test]
fn edge_counters_preserve_full_results_exact_profiles_and_final_ir() {
    assert!(!JitOptions::default().profile_edge_counters);
    assert!(JitOptions::default().verify_each_pass);
    let base_options = JitOptions::default();
    let ordinary = CpuJit::compile_with_options(SOURCE, base_options).unwrap();
    exercise(&ordinary, &UNSEEN, "ordinary independent oracle");
    let edge_ordinary = CpuJit::compile_with_options(
        SOURCE,
        JitOptions {
            profile_edge_counters: true,
            ..base_options
        },
    )
    .unwrap();
    assert_eq!(
        edge_ordinary.optimized_ir(),
        ordinary.optimized_ir(),
        "instrumentation option leaves ordinary IR identical"
    );
    exercise(
        &edge_ordinary,
        &UNSEEN,
        "edge option ordinary independent oracle",
    );

    let mut expected_profile = None;
    let mut expected_final_ir = None;
    for tier in [0, 1, 3] {
        let selected_options = JitOptions {
            training_opt_level: Some(tier),
            ..base_options
        };
        let edge_options = JitOptions {
            profile_edge_counters: true,
            ..selected_options
        };
        let select = CpuJit::compile_instrumented(SOURCE, selected_options).unwrap();
        let edge = CpuJit::compile_instrumented(SOURCE, edge_options).unwrap();
        let context = format!("edge instrumentation, training IR O{tier}, native/final O3");
        let empty = edge.branch_profile().unwrap();
        assert_eq!(
            empty.total_observations(),
            0,
            "{context}: eager compilation executes no source"
        );
        assert_eq!(empty, select.branch_profile().unwrap());
        assert!(edge.optimized_ir().contains("atomicrmw"));
        assert_eq!(edge.compile_timings().verification_checks, 2);

        let select_oracle = exercise(&select, &TRAINING, &context);
        let edge_oracle = exercise(&edge, &TRAINING, &context);
        let measured = edge.branch_profile().unwrap();
        select_oracle.check(&select.branch_profile().unwrap(), &context);
        edge_oracle.check(&measured, &context);
        assert_eq!(
            measured,
            select.branch_profile().unwrap(),
            "{context}: every site, outcome and original fingerprint"
        );
        if let Some(expected) = &expected_profile {
            assert_eq!(
                &measured, expected,
                "{context}: profile is tier-independent"
            );
        } else {
            expected_profile = Some(measured.clone());
        }
        assert_eq!(
            empty.total_observations(),
            0,
            "{context}: owned empty snapshot"
        );

        let final_select =
            CpuJit::compile_with_profile(SOURCE, selected_options, &measured).unwrap();
        let final_edge = CpuJit::compile_with_profile(SOURCE, edge_options, &measured).unwrap();
        assert_eq!(final_edge.compile_timings().verification_checks, 3);
        assert_eq!(
            final_edge.optimized_ir(),
            final_select.optimized_ir(),
            "{context}: training instrumentation leaves final IR identical"
        );
        if let Some(expected) = &expected_final_ir {
            assert_eq!(
                final_edge.optimized_ir(),
                expected,
                "{context}: final IR is tier-independent"
            );
        } else {
            expected_final_ir = Some(final_edge.optimized_ir().to_owned());
        }
        assert_eq!(
            final_edge.profiled_branches(),
            final_select.profiled_branches()
        );
        assert_eq!(
            final_edge.profile_selection_optimization(),
            final_select.profile_selection_optimization()
        );
        assert!(!final_edge.optimized_ir().contains("atomicrmw"));
        exercise(&final_edge, &UNSEEN, &context);
        assert_eq!(
            edge.branch_profile().unwrap(),
            measured,
            "{context}: final compilation and calls execute no trainer source"
        );

        // Both old native sessions remain callable after final compilation.
        // Repeat the same unseen inputs and independently check accumulated
        // choose/Fibonacci counts, rather than only checking an increasing sum.
        let before = measured.clone();
        let unseen_oracle = exercise(&edge, &UNSEEN, &context);
        exercise(&select, &UNSEEN, &context);
        let accumulated = edge.branch_profile().unwrap();
        assert_eq!(
            accumulated,
            select.branch_profile().unwrap(),
            "{context}: exact accumulated profile across unseen inputs"
        );
        let cumulative = CounterOracle {
            choose_true: edge_oracle.choose_true + unseen_oracle.choose_true,
            choose_false: edge_oracle.choose_false + unseen_oracle.choose_false,
            fib_true: edge_oracle.fib_true + unseen_oracle.fib_true,
            fib_false: edge_oracle.fib_false + unseen_oracle.fib_false,
        };
        cumulative.check(&accumulated, &context);
        assert!(accumulated.total_observations() > measured.total_observations());
        assert_eq!(measured, before, "{context}: old snapshot owns counts");
        drop(edge);
        drop(select);
        assert_eq!(
            measured,
            expected_profile.as_ref().unwrap().clone(),
            "{context}: profile survives both trainers"
        );
        exercise(&final_edge, &UNSEEN, &context);
    }
}

#[test]
fn edge_counters_preserve_atomic_native_concurrency_and_owned_snapshots() {
    use std::sync::{Arc, Barrier};

    const CONCURRENT_SOURCE: &str = r#"
fn choose(x: I64) -> I64 {
    if x < 0 { return x - 1; }
    return x + 2;
}
"#;
    for tier in [0, 1, 3] {
        let training = CpuJit::compile_instrumented(
            CONCURRENT_SOURCE,
            JitOptions {
                training_opt_level: Some(tier),
                profile_edge_counters: true,
                ..JitOptions::default()
            },
        )
        .unwrap();
        let empty = training.branch_profile().unwrap();
        assert_eq!(empty.total_observations(), 0);
        let address = training.function_address("choose").unwrap();
        let phase = Arc::new(Barrier::new(5));
        let threads: Vec<_> = (0..4)
            .map(|thread| {
                let phase = Arc::clone(&phase);
                std::thread::spawn(move || {
                    // Native callers share only the atomic profiling global;
                    // the creating thread owns the JIT until all callers join.
                    let choose: unsafe extern "C" fn(i64) -> i64 =
                        unsafe { std::mem::transmute(address) };
                    let mut wrong_results = 0;
                    phase.wait();
                    for i in 0..512 {
                        let x = if (thread + i) % 2 == 0 { -1 } else { 0 };
                        wrong_results +=
                            usize::from(unsafe { choose(x) } != if x < 0 { -2 } else { 2 });
                    }
                    phase.wait();
                    phase.wait();
                    for i in 512..4096 {
                        let x = if (thread + i) % 2 == 0 { -1 } else { 0 };
                        wrong_results +=
                            usize::from(unsafe { choose(x) } != if x < 0 { -2 } else { 2 });
                    }
                    wrong_results
                })
            })
            .collect();
        phase.wait();
        phase.wait();
        let paused = training.branch_profile().unwrap();
        // Release every worker before asserting, so a wrong native result or
        // count produces a failure without stranding peers at the barrier.
        phase.wait();
        assert_eq!(paused.sites().len(), 1);
        assert_eq!(
            (paused.sites()[0].true_count, paused.sites()[0].false_count),
            (1024, 1024),
            "O{tier}: completed concurrent calls are fully observed"
        );
        let during = training.branch_profile().unwrap();
        assert!(
            (1024..=8192).contains(&during.sites()[0].true_count)
                && (1024..=8192).contains(&during.sites()[0].false_count),
            "O{tier}: concurrent snapshot reads bounded atomic counts"
        );
        for thread in threads {
            assert_eq!(
                thread.join().unwrap(),
                0,
                "O{tier}: complete native results"
            );
        }
        let final_profile = training.branch_profile().unwrap();
        assert_eq!(
            (
                final_profile.sites()[0].true_count,
                final_profile.sites()[0].false_count
            ),
            (8192, 8192),
            "O{tier}: no lost concurrent outcomes"
        );
        assert_eq!(final_profile.total_observations(), 16384);
        assert_eq!(empty.total_observations(), 0);
        assert_eq!(paused.total_observations(), 2048);
        drop(training);
        assert_eq!(final_profile.total_observations(), 16384);
        assert_eq!(paused.total_observations(), 2048);
    }
}

#[test]
fn loop_edge_counters_preserve_independent_full_results_and_all_profiles() {
    assert!(!JitOptions::default().profile_loop_edge_counters);
    assert!(!JitOptions::default().profile_edge_counters);
    assert!(JitOptions::default().verify_each_pass);
    for codegen_opt_level in [None, Some(1)] {
        let base = JitOptions {
            codegen_opt_level,
            ..JitOptions::default()
        };
        let ordinary = CpuJit::compile_with_options(SOURCE, base).unwrap();
        exercise(
            &ordinary,
            &UNSEEN,
            "loop-option ordinary independent oracle",
        );
        let ordinary_loop = CpuJit::compile_with_options(
            SOURCE,
            JitOptions {
                profile_loop_edge_counters: true,
                ..base
            },
        )
        .unwrap();
        assert_eq!(
            ordinary_loop.optimized_ir(),
            ordinary.optimized_ir(),
            "loop instrumentation option leaves ordinary IR identical"
        );
        exercise(
            &ordinary_loop,
            &UNSEEN,
            "loop-option ordinary independent oracle",
        );

        let mut expected_profile = None;
        let mut expected_final_ir = None;
        for tier in 0..=3 {
            let selected_options = JitOptions {
                training_opt_level: Some(tier),
                ..base
            };
            let loop_options = JitOptions {
                profile_loop_edge_counters: true,
                ..selected_options
            };
            let selected = CpuJit::compile_instrumented(SOURCE, selected_options).unwrap();
            let training = CpuJit::compile_instrumented(SOURCE, loop_options).unwrap();
            let context = format!(
                "loop-only edges, training IR O{tier}, native override {codegen_opt_level:?}"
            );
            let empty = training.branch_profile().unwrap();
            assert_eq!(
                empty.total_observations(),
                0,
                "{context}: no implicit training"
            );
            assert_eq!(empty, selected.branch_profile().unwrap());
            assert!(training.optimized_ir().contains("atomicrmw"));
            assert_eq!(training.compile_timings().verification_checks, 2);

            let selected_oracle = exercise(&selected, &TRAINING, &context);
            let oracle = exercise(&training, &TRAINING, &context);
            let measured = training.branch_profile().unwrap();
            oracle.check(&measured, &context);
            selected_oracle.check(&selected.branch_profile().unwrap(), &context);
            assert_eq!(
                measured,
                selected.branch_profile().unwrap(),
                "{context}: every loop/work count, identity and fingerprint"
            );
            if let Some(expected) = &expected_profile {
                assert_eq!(&measured, expected, "{context}: exact cross-tier profile");
            } else {
                expected_profile = Some(measured.clone());
            }

            let final_selected =
                CpuJit::compile_with_profile(SOURCE, selected_options, &measured).unwrap();
            let final_loop = CpuJit::compile_with_profile(SOURCE, loop_options, &measured).unwrap();
            assert_eq!(final_loop.compile_timings().verification_checks, 3);
            assert_eq!(
                final_loop.optimized_ir(),
                final_selected.optimized_ir(),
                "{context}: final IR ignores loop instrumentation option"
            );
            assert!(!final_loop.optimized_ir().contains("atomicrmw"));
            assert_eq!(
                final_loop.profiled_branches(),
                final_selected.profiled_branches()
            );
            assert_eq!(
                final_loop.profile_selection_optimization(),
                final_selected.profile_selection_optimization()
            );
            if let Some(expected) = &expected_final_ir {
                assert_eq!(
                    final_loop.optimized_ir(),
                    expected,
                    "{context}: final IR ignores training tier"
                );
            } else {
                expected_final_ir = Some(final_loop.optimized_ir().to_owned());
            }
            exercise(&final_loop, &UNSEEN, &context);
            assert_eq!(training.branch_profile().unwrap(), measured);

            // The flags form a union. Check both flags against all-edge mode
            // under a separate intermediate IR/native combination as well.
            if tier == 2 && codegen_opt_level == Some(1) {
                let all_options = JitOptions {
                    profile_edge_counters: true,
                    ..selected_options
                };
                let both_options = JitOptions {
                    profile_loop_edge_counters: true,
                    ..all_options
                };
                let all = CpuJit::compile_instrumented(SOURCE, all_options).unwrap();
                let both = CpuJit::compile_instrumented(SOURCE, both_options).unwrap();
                assert_eq!(
                    both.optimized_ir(),
                    all.optimized_ir(),
                    "both flags select the established all-edge instrumented IR"
                );
                let both_oracle = exercise(&both, &TRAINING, &context);
                exercise(&all, &TRAINING, &context);
                both_oracle.check(&both.branch_profile().unwrap(), &context);
                assert_eq!(both.branch_profile().unwrap(), measured);
                assert_eq!(all.branch_profile().unwrap(), measured);
            }

            let before = measured.clone();
            let unseen = exercise(&training, &UNSEEN, &context);
            exercise(&selected, &UNSEEN, &context);
            let accumulated = training.branch_profile().unwrap();
            assert_eq!(accumulated, selected.branch_profile().unwrap());
            let cumulative = CounterOracle {
                choose_true: oracle.choose_true + unseen.choose_true,
                choose_false: oracle.choose_false + unseen.choose_false,
                fib_true: oracle.fib_true + unseen.fib_true,
                fib_false: oracle.fib_false + unseen.fib_false,
            };
            cumulative.check(&accumulated, &context);
            assert!(accumulated.total_observations() > measured.total_observations());
            assert_eq!(
                measured, before,
                "{context}: snapshot owns its original counts"
            );
            assert_eq!(empty.total_observations(), 0);
            drop(training);
            drop(selected);
            assert_eq!(measured, expected_profile.as_ref().unwrap().clone());
            exercise(&final_loop, &UNSEEN, &context);
        }
    }
}

#[derive(Clone, Copy, Default)]
struct NativeLoopOracle {
    loop_true: u64,
    loop_false: u64,
    work_true: u64,
    work_false: u64,
}

impl NativeLoopOracle {
    fn observe(&mut self, n: i64, salt: i64) -> i64 {
        let length = n.max(0) as u64;
        self.loop_true += length;
        self.loop_false += 1;
        let even_salt = salt & 1 == 0;
        let yes = if even_salt {
            length.div_ceil(2)
        } else {
            length / 2
        };
        self.work_true += yes;
        self.work_false += length - yes;
        // Alternating +/- 1..n has a closed-form result. This oracle does
        // not repeat the source loop or its branch transitions.
        let alternating = if length & 1 == 0 {
            -(length as i64 / 2)
        } else {
            length as i64 / 2 + 1
        };
        if even_salt {
            alternating
        } else {
            -alternating
        }
    }

    fn expected_profile(self) -> [(u64, u64); 2] {
        [
            (self.loop_true, self.loop_false),
            (self.work_true, self.work_false),
        ]
    }

    fn check(self, profile: &BranchProfile, context: &str) {
        assert_eq!(
            profile.sites().len(),
            2,
            "{context}: loop and interior work"
        );
        assert!(profile.sites()[0].block.starts_with("while.cond."));
        assert!(profile.sites()[1].block.starts_with("while.body."));
        for (site, (yes, no)) in profile.sites().iter().zip(self.expected_profile()) {
            assert_eq!(site.function, "loop_work");
            assert_eq!(
                (site.true_count, site.false_count),
                (yes, no),
                "{context}: independently counted {}",
                site.block
            );
        }
    }
}

#[test]
fn loop_edge_counters_observe_native_loop_and_work_concurrency_exactly() {
    use std::sync::{Arc, Barrier};

    const CONCURRENT_LOOP_SOURCE: &str = r#"
@unsafe
fn loop_work(n: I64, salt: I64) -> I64 {
    let mut i: I64 = 0;
    let mut sum: I64 = 0;
    while i < n {
        if ((i + salt) & 1) == 0 { sum = sum + i + 1; }
        else { sum = sum - i - 1; }
        i = i + 1;
    }
    return sum;
}
"#;
    for codegen_opt_level in [None, Some(1)] {
        for tier in 0..=3 {
            let context = format!("loop concurrency IR O{tier}, native {codegen_opt_level:?}");
            let options = JitOptions {
                training_opt_level: Some(tier),
                codegen_opt_level,
                profile_loop_edge_counters: true,
                ..JitOptions::default()
            };
            let training = CpuJit::compile_instrumented(CONCURRENT_LOOP_SOURCE, options).unwrap();
            let empty = training.branch_profile().unwrap();
            assert_eq!(empty.total_observations(), 0);
            let address = training.function_address("loop_work").unwrap();
            let phase = Arc::new(Barrier::new(5));
            let threads: Vec<_> = (0..4)
                .map(|thread| {
                    let phase = Arc::clone(&phase);
                    std::thread::spawn(move || {
                        // The parent owns the live JIT; source values are
                        // scalar and only profiling atomics are shared.
                        let native: unsafe extern "C" fn(i64, i64) -> i64 =
                            unsafe { std::mem::transmute(address) };
                        let mut oracle = NativeLoopOracle::default();
                        let mut wrong_results = 0;
                        phase.wait();
                        for i in 0..512 {
                            let n = ((thread + i) % 10) as i64 - 1;
                            let salt = (thread + i) as i64;
                            wrong_results +=
                                usize::from(unsafe { native(n, salt) } != oracle.observe(n, salt));
                        }
                        phase.wait();
                        phase.wait();
                        for i in 512..4096 {
                            let n = ((thread + i) % 10) as i64 - 1;
                            let salt = (thread + i) as i64;
                            wrong_results +=
                                usize::from(unsafe { native(n, salt) } != oracle.observe(n, salt));
                        }
                        wrong_results
                    })
                })
                .collect();
            let mut paused_oracle = NativeLoopOracle::default();
            let mut final_oracle = NativeLoopOracle::default();
            for thread in 0..4 {
                for i in 0..4096 {
                    let n = ((thread + i) % 10) as i64 - 1;
                    let salt = (thread + i) as i64;
                    final_oracle.observe(n, salt);
                    if i < 512 {
                        paused_oracle.observe(n, salt);
                    }
                }
            }
            phase.wait();
            phase.wait();
            let paused = training.branch_profile().unwrap();
            phase.wait();
            paused_oracle.check(&paused, &context);
            let during = training.branch_profile().unwrap();
            for ((site, (minimum_true, minimum_false)), (maximum_true, maximum_false)) in during
                .sites()
                .iter()
                .zip(paused_oracle.expected_profile())
                .zip(final_oracle.expected_profile())
            {
                assert!(
                    (minimum_true..=maximum_true).contains(&site.true_count)
                        && (minimum_false..=maximum_false).contains(&site.false_count),
                    "{context}: bounded concurrent {} snapshot",
                    site.block
                );
            }
            for thread in threads {
                assert_eq!(thread.join().unwrap(), 0, "{context}: all native results");
            }
            let final_profile = training.branch_profile().unwrap();
            final_oracle.check(&final_profile, &context);
            assert!(final_profile.total_observations() > paused.total_observations());
            paused_oracle.check(&paused, &context);
            assert_eq!(empty.total_observations(), 0);
            drop(training);
            final_oracle.check(&final_profile, &context);
            paused_oracle.check(&paused, &context);
        }
    }
}
