//! Final-loop-unrolling policy must preserve complete source outcomes.
//! Independent bounded oracles cover full buffers, runtime object contents,
//! IEEE behavior, recursion and guarded side effects. Instrumented modules and
//! exact original-site profiles are unchanged between final-policy arms.
//! These correctness gates are not arbitrary-program equivalence proofs.
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

fn check_compilation_contract(jit: &CpuJit, instrumented: bool, expected_checks: u32) {
    assert_eq!(jit.compile_timings().verification_checks, expected_checks);
    let materialization = jit.materialization_timings();
    assert_eq!(
        materialization.function_lookup_count,
        2 * jit.functions().count() as u64,
        "every source function and checked adapter is materialized eagerly"
    );
    assert_eq!(
        materialization.profile_lookup_count,
        u64::from(instrumented)
    );
    assert!(materialization.first_lookup > Duration::ZERO);
    assert!(materialization.submission > Duration::ZERO);
    if materialization.object_observer_available {
        assert_eq!(materialization.object_count, 1);
        assert!(materialization.object_bytes > 0);
    }
    let snapshot = *materialization;
    for name in jit.functions() {
        assert_ne!(jit.function_address(name).unwrap(), 0);
    }
    assert_eq!(*jit.materialization_timings(), snapshot);
}

#[test]
fn final_unrolling_policy_preserves_independent_full_results_and_training_identity() {
    assert!(JitOptions::default().final_loop_unrolling);
    assert!(JitOptions::default().verify_each_pass);
    for opt_level in 0..=3 {
        for codegen_opt_level in [None, Some(1)] {
            let defaults = JitOptions {
                opt_level,
                codegen_opt_level,
                ..JitOptions::default()
            };
            let disabled = JitOptions {
                final_loop_unrolling: false,
                ..defaults
            };
            let context = format!("final IR O{opt_level}, native override {codegen_opt_level:?}");
            let ordinary_default = CpuJit::compile_with_options(SOURCE, defaults).unwrap();
            let ordinary_disabled = CpuJit::compile_with_options(SOURCE, disabled).unwrap();
            for ordinary in [&ordinary_default, &ordinary_disabled] {
                check_compilation_contract(ordinary, false, 2);
                exercise(ordinary, &UNSEEN, &context);
                assert!(!ordinary.optimized_ir().contains("atomicrmw"));
            }

            let training_default = CpuJit::compile_instrumented(SOURCE, defaults).unwrap();
            let training_disabled = CpuJit::compile_instrumented(SOURCE, disabled).unwrap();
            check_compilation_contract(&training_default, true, 2);
            check_compilation_contract(&training_disabled, true, 2);
            assert_eq!(
                training_disabled.optimized_ir(),
                training_default.optimized_ir(),
                "{context}: final unrolling policy leaves training IR identical"
            );
            let empty = training_disabled.branch_profile().unwrap();
            assert_eq!(empty.total_observations(), 0);
            assert_eq!(empty, training_default.branch_profile().unwrap());
            let default_oracle = exercise(&training_default, &TRAINING, &context);
            let oracle = exercise(&training_disabled, &TRAINING, &context);
            let measured = training_disabled.branch_profile().unwrap();
            default_oracle.check(&training_default.branch_profile().unwrap(), &context);
            oracle.check(&measured, &context);
            assert_eq!(
                measured,
                training_default.branch_profile().unwrap(),
                "{context}: all original profiles/counts/fingerprint identical"
            );

            let profiled_default =
                CpuJit::compile_with_profile(SOURCE, defaults, &measured).unwrap();
            let profiled_disabled =
                CpuJit::compile_with_profile(SOURCE, disabled, &measured).unwrap();
            let expected_checks = if opt_level == 0 { 2 } else { 3 };
            for profiled in [&profiled_default, &profiled_disabled] {
                check_compilation_contract(profiled, false, expected_checks);
                assert!(!profiled.optimized_ir().contains("atomicrmw"));
                exercise(profiled, &UNSEEN, &context);
            }
            assert_eq!(
                profiled_disabled.profiled_branches(),
                profiled_default.profiled_branches()
            );
            assert_eq!(
                profiled_disabled.profile_selection_optimization(),
                profiled_default.profile_selection_optimization()
            );
            assert_eq!(training_disabled.branch_profile().unwrap(), measured);
            assert_eq!(training_default.branch_profile().unwrap(), measured);

            let owned_before = measured.clone();
            let unseen = exercise(&training_disabled, &UNSEEN, &context);
            exercise(&training_default, &UNSEEN, &context);
            let accumulated = training_disabled.branch_profile().unwrap();
            assert_eq!(accumulated, training_default.branch_profile().unwrap());
            CounterOracle {
                choose_true: oracle.choose_true + unseen.choose_true,
                choose_false: oracle.choose_false + unseen.choose_false,
                fib_true: oracle.fib_true + unseen.fib_true,
                fib_false: oracle.fib_false + unseen.fib_false,
            }
            .check(&accumulated, &context);
            assert!(accumulated.total_observations() > measured.total_observations());
            assert_eq!(empty.total_observations(), 0);
            assert_eq!(measured, owned_before);
            drop(training_disabled);
            drop(training_default);
            oracle.check(&measured, &context);
            exercise(&profiled_disabled, &UNSEEN, &context);
            exercise(&profiled_default, &UNSEEN, &context);
        }
    }
}

const COUNTED_SOURCE: &str = r#"
@unsafe
fn counted(n: I64, seed: I64) -> I64 {
    let mut i: I64 = 0;
    let mut value: I64 = seed;
    while i < n {
        value = (value * 48271) % 2147483647;
        i = i + 1;
    }
    return value;
}
"#;

fn counted_oracle(n: i64, seed: i64) -> i64 {
    let mut value = i128::from(seed);
    for _ in 0..n.max(0) {
        value = (value * 48271) % 2147483647;
    }
    value as i64
}

fn counted_function(ir: &str) -> &str {
    let definition = ir
        .lines()
        .find(|line| line.starts_with("define ") && line.contains("@counted("))
        .unwrap();
    let start = ir.find(definition).unwrap();
    let function = &ir[start..];
    &function[..function.find("\n}").unwrap() + 2]
}

#[test]
fn final_unrolling_false_reaches_ordinary_and_profile_use_pipelines() {
    let defaults = JitOptions::default();
    let disabled = JitOptions {
        final_loop_unrolling: false,
        ..defaults
    };
    let training = CpuJit::compile_instrumented(COUNTED_SOURCE, defaults).unwrap();
    for n in [0, 1, 2, 31, 129] {
        assert_eq!(
            call(
                &training,
                "counted",
                &[V::I64(n), V::I64(1)],
                "counted training",
            ),
            V::I64(counted_oracle(n, 1))
        );
    }
    let profile = training.branch_profile().unwrap();
    let variants = [
        (
            CpuJit::compile_with_options(COUNTED_SOURCE, defaults).unwrap(),
            CpuJit::compile_with_options(COUNTED_SOURCE, disabled).unwrap(),
        ),
        (
            CpuJit::compile_with_profile(COUNTED_SOURCE, defaults, &profile).unwrap(),
            CpuJit::compile_with_profile(COUNTED_SOURCE, disabled, &profile).unwrap(),
        ),
    ];
    for (default, disabled) in variants {
        let default_function = counted_function(default.optimized_ir());
        let disabled_function = counted_function(disabled.optimized_ir());
        // This nonlinear scalar recurrence has no vector/closed-form replacement
        // in the measured LLVM baseline. Default O3 duplicates its recurrence
        // body; the disabled unrolling policy retains one recurrence multiply.
        // No assertion restricts vectorization or unrelated loop simplification.
        let default_multiplies = default_function.matches("mul ").count();
        let disabled_multiplies = disabled_function.matches("mul ").count();
        assert_eq!(disabled_multiplies, 1, "{disabled_function}");
        assert!(
            default_multiplies > disabled_multiplies,
            "known O3 counted-loop fixture must exercise unrolling\n{default_function}"
        );
        assert_ne!(default_function, disabled_function);
        for n in [-1, 0, 1, 2, 31, 129] {
            for seed in [-19, 0, 1, 2147483646] {
                let expected = V::I64(counted_oracle(n, seed));
                for jit in [&default, &disabled] {
                    assert_eq!(
                        call(
                            jit,
                            "counted",
                            &[V::I64(n), V::I64(seed)],
                            "unseen counted input"
                        ),
                        expected
                    );
                }
            }
        }
    }
    assert_eq!(training.branch_profile().unwrap(), profile);
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
fn final_unrolling_flag_preserves_native_loop_and_work_snapshot_concurrency() {
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
            for final_loop_unrolling in [true, false] {
                let context = format!("unrolling={final_loop_unrolling}, trainingIR O{tier}, native {codegen_opt_level:?}");
                let options = JitOptions {
                    training_opt_level: Some(tier),
                    codegen_opt_level,
                    final_loop_unrolling,
                    profile_loop_edge_counters: tier % 2 != 0,
                    ..JitOptions::default()
                };
                let training =
                    CpuJit::compile_instrumented(CONCURRENT_LOOP_SOURCE, options).unwrap();
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
                                wrong_results += usize::from(
                                    unsafe { native(n, salt) } != oracle.observe(n, salt),
                                );
                            }
                            phase.wait();
                            phase.wait();
                            for i in 512..4096 {
                                let n = ((thread + i) % 10) as i64 - 1;
                                let salt = (thread + i) as i64;
                                wrong_results += usize::from(
                                    unsafe { native(n, salt) } != oracle.observe(n, salt),
                                );
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
}
