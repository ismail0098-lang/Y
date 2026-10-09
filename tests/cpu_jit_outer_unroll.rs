//! Outer-loop unrolling policy must preserve complete source outcomes.
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

const NESTED_SOURCE: &str = r#"
@unsafe
fn nested(output: GlobalMemory<I64>, outer: I64, inner: I64, salt: I64) -> I64 {
    let mut i: I64 = 0;
    let mut sum: I64 = 0;
    while i < outer {
        let mut j: I64 = 0;
        while j < inner {
            let index: I64 = (i * 7 + j) % 16;
            if ((i + j + salt) & 1) == 0 {
                output[index] = output[index] + i + j + 1;
                sum = sum + output[index];
            } else {
                output[index] = output[index] - i - j - 1;
                sum = sum - output[index];
            }
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}
"#;

const NESTED_TRAINING: [(i64, i64, i64); 4] = [(-1, 0, -2), (0, 5, 1), (3, 4, 2), (5, 3, -1)];
const NESTED_UNSEEN: [(i64, i64, i64); 6] = [
    (1, -1, 2),
    (2, 0, 1),
    (1, 1, -1),
    (2, 3, 2),
    (3, 7, 0),
    (7, 9, -3),
];

#[derive(Clone, Copy, Default)]
struct NestedOracle {
    outer_true: u64,
    outer_false: u64,
    inner_true: u64,
    inner_false: u64,
    work_true: u64,
    work_false: u64,
}

impl NestedOracle {
    fn observe(&mut self, outer: i64, inner: i64, salt: i64) -> i64 {
        let rows = outer.max(0) as u64;
        let columns = inner.max(0) as u64;
        let even_rows = rows.div_ceil(2);
        let matching_rows = if salt & 1 == 0 { even_rows } else { rows / 2 };
        let yes = rows * (columns / 2) + (columns & 1) * matching_rows;
        self.outer_true += rows;
        self.outer_false += 1;
        self.inner_true += rows * columns;
        self.inner_false += rows;
        self.work_true += yes;
        self.work_false += rows * columns - yes;

        // Closed-form alternating row sums for the scalar concurrency kernel.
        // The memory oracle below independently applies all state transitions.
        let row_sum = if columns & 1 == 0 {
            -(columns as i64 / 2) * (rows as i64 & 1)
        } else if rows & 1 == 0 {
            -7 * (rows as i64 / 2)
        } else {
            7 * (rows as i64 / 2) + 1 + columns as i64 / 2
        };
        if salt & 1 == 0 {
            row_sum
        } else {
            -row_sum
        }
    }

    fn expected(self) -> [(u64, u64); 3] {
        [
            (self.outer_true, self.outer_false),
            (self.inner_true, self.inner_false),
            (self.work_true, self.work_false),
        ]
    }

    fn check(self, profile: &BranchProfile, function: &str, context: &str) {
        let sites: Vec<_> = profile
            .sites()
            .iter()
            .filter(|s| s.function == function)
            .collect();
        assert_eq!(sites.len(), 3, "{context}: nested original loop/work sites");
        assert!(sites[0].block.starts_with("while.cond."));
        assert!(sites[1].block.starts_with("while.cond."));
        for (site, expected) in sites.iter().zip(self.expected()) {
            assert_eq!(
                (site.true_count, site.false_count),
                expected,
                "{context}: independently counted {}",
                site.block
            );
        }
    }
}

fn exercise_nested(
    jit: &CpuJit,
    cases: &[(i64, i64, i64)],
    oracle: &mut NestedOracle,
    context: &str,
) {
    let timings = *jit.compile_timings();
    for &(outer, inner, salt) in cases {
        oracle.observe(outer, inner, salt);
        let mut actual = std::array::from_fn::<_, 18, _>(|index| (index as i64 - 8) * 11);
        let mut expected = actual;
        let mut sum = 0;
        for i in 0..outer.max(0) {
            for j in 0..inner.max(0) {
                let index = 1 + ((i * 7 + j) % 16) as usize;
                if (i + j + salt) & 1 == 0 {
                    expected[index] += i + j + 1;
                    sum += expected[index];
                } else {
                    expected[index] -= i + j + 1;
                    sum -= expected[index];
                }
            }
        }
        assert_eq!(
            call(
                jit,
                "nested",
                &[
                    V::Pointer(unsafe { actual.as_mut_ptr().add(1) }.cast()),
                    V::I64(outer),
                    V::I64(inner),
                    V::I64(salt),
                ],
                context
            ),
            V::I64(sum),
            "{context}: complete nested result"
        );
        assert_eq!(
            actual, expected,
            "{context}: all nested memory words and canaries"
        );
    }
    assert_eq!(*jit.compile_timings(), timings);
}

#[test]
fn outer_unrolling_policy_preserves_full_contents_ieee_and_exact_training_profiles() {
    assert!(JitOptions::default().final_unroll_outer_loops);
    assert!(JitOptions::default().final_loop_unrolling);
    assert!(JitOptions::default().verify_each_pass);
    let source = format!("{SOURCE}\n{NESTED_SOURCE}");
    for opt_level in 0..=3 {
        for codegen_opt_level in [None, Some(1)] {
            let defaults = JitOptions {
                opt_level,
                codegen_opt_level,
                training_opt_level: Some((opt_level + 1) % 4),
                profile_loop_edge_counters: opt_level % 2 != 0,
                ..JitOptions::default()
            };
            let outer_disabled = JitOptions {
                final_unroll_outer_loops: false,
                ..defaults
            };
            let context = format!("outer-only, finalIR O{opt_level}, native {codegen_opt_level:?}");
            let ordinary_default = CpuJit::compile_with_options(&source, defaults).unwrap();
            let ordinary_disabled = CpuJit::compile_with_options(&source, outer_disabled).unwrap();
            assert_eq!(ordinary_default.outer_unroll_annotations(), 0);
            assert_eq!(ordinary_disabled.outer_unroll_annotations(), 1);
            for jit in [&ordinary_default, &ordinary_disabled] {
                check_compilation_contract(jit, false, 2);
                exercise(jit, &UNSEEN, &context);
                exercise_nested(jit, &NESTED_UNSEEN, &mut NestedOracle::default(), &context);
            }

            let training_default = CpuJit::compile_instrumented(&source, defaults).unwrap();
            let training_disabled = CpuJit::compile_instrumented(&source, outer_disabled).unwrap();
            for jit in [&training_default, &training_disabled] {
                check_compilation_contract(jit, true, 2);
                assert_eq!(jit.outer_unroll_annotations(), 0);
            }
            assert_eq!(
                training_default.optimized_ir(),
                training_disabled.optimized_ir(),
                "{context}: exact instrumented IR identity"
            );
            let empty = training_disabled.branch_profile().unwrap();
            assert_eq!(empty.total_observations(), 0);
            assert_eq!(empty, training_default.branch_profile().unwrap());
            let default_scalar = exercise(&training_default, &TRAINING, &context);
            let scalar = exercise(&training_disabled, &TRAINING, &context);
            let mut nested_default = NestedOracle::default();
            let mut nested = NestedOracle::default();
            exercise_nested(
                &training_default,
                &NESTED_TRAINING,
                &mut nested_default,
                &context,
            );
            exercise_nested(&training_disabled, &NESTED_TRAINING, &mut nested, &context);
            let measured = training_disabled.branch_profile().unwrap();
            assert_eq!(measured, training_default.branch_profile().unwrap());
            default_scalar.check(&measured, &context);
            scalar.check(&measured, &context);
            nested_default.check(&measured, "nested", &context);
            nested.check(&measured, "nested", &context);

            let profiled_default =
                CpuJit::compile_with_profile(&source, defaults, &measured).unwrap();
            let profiled_disabled =
                CpuJit::compile_with_profile(&source, outer_disabled, &measured).unwrap();
            assert_eq!(profiled_default.outer_unroll_annotations(), 0);
            assert_eq!(profiled_disabled.outer_unroll_annotations(), 1);
            for jit in [&profiled_default, &profiled_disabled] {
                check_compilation_contract(jit, false, if opt_level == 0 { 2 } else { 3 });
                assert!(!jit.optimized_ir().contains("atomicrmw"));
                exercise(jit, &UNSEEN, &context);
                exercise_nested(jit, &NESTED_UNSEEN, &mut NestedOracle::default(), &context);
            }
            assert_eq!(
                profiled_default.profiled_branches(),
                profiled_disabled.profiled_branches()
            );
            assert_eq!(
                profiled_default.profile_selection_optimization(),
                profiled_disabled.profile_selection_optimization()
            );
            assert_eq!(training_disabled.branch_profile().unwrap(), measured);
            let owned = measured.clone();
            let unseen = exercise(&training_disabled, &UNSEEN, &context);
            exercise(&training_default, &UNSEEN, &context);
            exercise_nested(&training_disabled, &NESTED_UNSEEN, &mut nested, &context);
            exercise_nested(
                &training_default,
                &NESTED_UNSEEN,
                &mut nested_default,
                &context,
            );
            let accumulated = training_disabled.branch_profile().unwrap();
            assert_eq!(accumulated, training_default.branch_profile().unwrap());
            CounterOracle {
                choose_true: scalar.choose_true + unseen.choose_true,
                choose_false: scalar.choose_false + unseen.choose_false,
                fib_true: scalar.fib_true + unseen.fib_true,
                fib_false: scalar.fib_false + unseen.fib_false,
            }
            .check(&accumulated, &context);
            nested.check(&accumulated, "nested", &context);
            assert!(accumulated.total_observations() > measured.total_observations());
            assert_eq!(owned, measured);
            assert_eq!(empty.total_observations(), 0);
            drop(training_default);
            drop(training_disabled);
            scalar.check(&measured, &context);
            exercise(&profiled_disabled, &UNSEEN, &context);
            exercise_nested(
                &profiled_disabled,
                &NESTED_UNSEEN,
                &mut NestedOracle::default(),
                &context,
            );
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

const NESTED_COUNTED_SOURCE: &str = r#"
@unsafe
fn nested_counted(seed: I64) -> I64 {
    let mut i: I64 = 0;
    let mut value: I64 = seed;
    while i < 4 {
        let mut j: I64 = 0;
        while j < 3 {
            value = (value * 48271) % 2147483647;
            j = j + 1;
        }
        i = i + 1;
    }
    return value;
}
"#;

fn recurrence_oracle(n: i64, seed: i64) -> i64 {
    let mut value = i128::from(seed);
    for _ in 0..n.max(0) {
        value = (value * 48271) % 2147483647;
    }
    value as i64
}

fn source_function<'a>(ir: &'a str, name: &str) -> &'a str {
    let definition = ir
        .lines()
        .find(|line| line.starts_with("define ") && line.contains(&format!("@{name}(")))
        .unwrap();
    let tail = &ir[ir.find(definition).unwrap()..];
    &tail[..tail.find("\n}").unwrap() + 2]
}

#[test]
fn outer_policy_metadata_preserves_inner_tuning_and_changes_known_o3_nested_recurrence() {
    let defaults = JitOptions::default();
    let outer_disabled = JitOptions {
        final_unroll_outer_loops: false,
        ..defaults
    };
    let all_disabled = JitOptions {
        final_loop_unrolling: false,
        ..outer_disabled
    };
    let training = CpuJit::compile_instrumented(NESTED_COUNTED_SOURCE, defaults).unwrap();
    for seed in [-19, 0, 1, 2147483646] {
        assert_eq!(
            call(
                &training,
                "nested_counted",
                &[V::I64(seed)],
                "nested training"
            ),
            V::I64(recurrence_oracle(12, seed))
        );
    }
    let profile = training.branch_profile().unwrap();
    let zero = JitOptions {
        opt_level: 0,
        verify_each_pass: false,
        ..outer_disabled
    };
    let o0 = CpuJit::compile_with_options(NESTED_COUNTED_SOURCE, zero).unwrap();
    let o0_profiled = CpuJit::compile_with_profile(NESTED_COUNTED_SOURCE, zero, &profile).unwrap();
    for jit in [&o0, &o0_profiled] {
        assert_eq!(jit.outer_unroll_annotations(), 1);
        check_compilation_contract(jit, false, 2);
        assert_eq!(
            jit.optimized_ir()
                .matches("!\"llvm.loop.unroll.disable\"")
                .count(),
            1
        );
        assert_eq!(
            source_function(jit.optimized_ir(), "nested_counted")
                .matches("!llvm.loop")
                .count(),
            1,
            "only the outer latch receives original-CFG metadata at O0"
        );
        assert!(jit.optimized_ir().contains("distinct !{"));
    }
    let ordinary = [
        CpuJit::compile_with_options(NESTED_COUNTED_SOURCE, defaults).unwrap(),
        CpuJit::compile_with_options(NESTED_COUNTED_SOURCE, outer_disabled).unwrap(),
        CpuJit::compile_with_options(NESTED_COUNTED_SOURCE, all_disabled).unwrap(),
    ];
    let profiled = [
        CpuJit::compile_with_profile(NESTED_COUNTED_SOURCE, defaults, &profile).unwrap(),
        CpuJit::compile_with_profile(NESTED_COUNTED_SOURCE, outer_disabled, &profile).unwrap(),
        CpuJit::compile_with_profile(NESTED_COUNTED_SOURCE, all_disabled, &profile).unwrap(),
    ];
    for variants in [&ordinary, &profiled] {
        assert_eq!(variants[0].outer_unroll_annotations(), 0);
        assert_eq!(variants[1].outer_unroll_annotations(), 1);
        assert_eq!(variants[2].outer_unroll_annotations(), 0);
        let functions: Vec<_> = variants
            .iter()
            .map(|jit| source_function(jit.optimized_ir(), "nested_counted"))
            .collect();
        let multiples: Vec<_> = functions
            .iter()
            .map(|body| body.matches("mul ").count())
            .collect();
        assert!(
            multiples[0] > multiples[1],
            "known default O3 outer expansion must be reduced: {multiples:?}\n{functions:?}"
        );
        assert!(
            multiples[1] > multiples[2],
            "inner-loop default unrolling must remain enabled: {multiples:?}\n{functions:?}"
        );
        assert_eq!(multiples[2], 1);
        for seed in [-19, 0, 1, 2147483646] {
            for jit in variants {
                assert_eq!(
                    call(jit, "nested_counted", &[V::I64(seed)], "nested recurrence"),
                    V::I64(recurrence_oracle(12, seed))
                );
            }
        }
    }
    assert_eq!(training.branch_profile().unwrap(), profile);
}

#[test]
fn global_disable_takes_precedence_and_single_loops_keep_identical_final_ir() {
    for opt_level in 0..=3 {
        let defaults = JitOptions {
            opt_level,
            codegen_opt_level: Some(1),
            ..JitOptions::default()
        };
        let outer_disabled = JitOptions {
            final_unroll_outer_loops: false,
            ..defaults
        };
        let training = CpuJit::compile_instrumented(COUNTED_SOURCE, defaults).unwrap();
        for n in [0, 1, 2, 31, 129] {
            assert_eq!(
                call(
                    &training,
                    "counted",
                    &[V::I64(n), V::I64(1)],
                    "single training"
                ),
                V::I64(recurrence_oracle(n, 1))
            );
        }
        let profile = training.branch_profile().unwrap();
        let pairs = [
            (
                CpuJit::compile_with_options(COUNTED_SOURCE, defaults).unwrap(),
                CpuJit::compile_with_options(COUNTED_SOURCE, outer_disabled).unwrap(),
            ),
            (
                CpuJit::compile_with_profile(COUNTED_SOURCE, defaults, &profile).unwrap(),
                CpuJit::compile_with_profile(COUNTED_SOURCE, outer_disabled, &profile).unwrap(),
            ),
        ];
        for (default, outer) in pairs {
            assert_eq!(
                default.optimized_ir(),
                outer.optimized_ir(),
                "single loop finalIR O{opt_level}"
            );
            assert_eq!(outer.outer_unroll_annotations(), 0);
            for n in [-1, 0, 1, 2, 3, 31, 129] {
                for seed in [-19, 0, 1, 2147483646] {
                    for jit in [&default, &outer] {
                        assert_eq!(
                            call(jit, "counted", &[V::I64(n), V::I64(seed)], "single unseen"),
                            V::I64(recurrence_oracle(n, seed))
                        );
                    }
                }
            }
        }
        let global = JitOptions {
            final_loop_unrolling: false,
            ..defaults
        };
        let global_outer = JitOptions {
            final_unroll_outer_loops: false,
            ..global
        };
        let nested_training = CpuJit::compile_instrumented(NESTED_COUNTED_SOURCE, global).unwrap();
        call(
            &nested_training,
            "nested_counted",
            &[V::I64(1)],
            "global training",
        );
        let nested_profile = nested_training.branch_profile().unwrap();
        let pairs = [
            (
                CpuJit::compile_with_options(NESTED_COUNTED_SOURCE, global).unwrap(),
                CpuJit::compile_with_options(NESTED_COUNTED_SOURCE, global_outer).unwrap(),
            ),
            (
                CpuJit::compile_with_profile(NESTED_COUNTED_SOURCE, global, &nested_profile)
                    .unwrap(),
                CpuJit::compile_with_profile(NESTED_COUNTED_SOURCE, global_outer, &nested_profile)
                    .unwrap(),
            ),
        ];
        for (global, both) in pairs {
            assert_eq!(
                global.optimized_ir(),
                both.optimized_ir(),
                "global precedence finalIR O{opt_level}"
            );
            assert_eq!(global.outer_unroll_annotations(), 0);
            assert_eq!(both.outer_unroll_annotations(), 0);
            assert!(!both
                .optimized_ir()
                .contains("!\"llvm.loop.unroll.disable\""));
            assert_eq!(
                call(&both, "nested_counted", &[V::I64(-19)], "global precedence"),
                V::I64(recurrence_oracle(12, -19))
            );
        }
    }
}

const CONCURRENT_NESTED_SOURCE: &str = r#"
@unsafe
fn nested_work(outer: I64, inner: I64, salt: I64) -> I64 {
    let mut i: I64 = 0;
    let mut sum: I64 = 0;
    while i < outer {
        let mut j: I64 = 0;
        while j < inner {
            if ((i + j + salt) & 1) == 0 { sum = sum + i * 7 + j + 1; }
            else { sum = sum - i * 7 - j - 1; }
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}
"#;

fn native_nested_address(jit: &CpuJit) -> usize {
    jit.function_address("nested_work").unwrap()
}

fn concurrent_arguments(thread: usize, i: usize) -> (i64, i64, i64) {
    (
        ((thread + i) % 8) as i64 - 1,
        ((thread * 3 + i) % 9) as i64 - 1,
        (thread + i) as i64,
    )
}

#[test]
fn outer_policy_preserves_native_nested_concurrency_and_owned_exact_snapshots() {
    use std::sync::{Arc, Barrier};
    for codegen_opt_level in [None, Some(1)] {
        for tier in 0..=3 {
            for final_unroll_outer_loops in [true, false] {
                let options = JitOptions {
                    training_opt_level: Some(tier),
                    codegen_opt_level,
                    final_unroll_outer_loops,
                    profile_loop_edge_counters: tier % 2 != 0,
                    profile_edge_counters: tier == 2,
                    ..JitOptions::default()
                };
                let context = format!("outer={final_unroll_outer_loops}, trainingIR O{tier}, native {codegen_opt_level:?}");
                let training =
                    CpuJit::compile_instrumented(CONCURRENT_NESTED_SOURCE, options).unwrap();
                assert_eq!(training.outer_unroll_annotations(), 0);
                let empty = training.branch_profile().unwrap();
                let address = native_nested_address(&training);
                let phase = Arc::new(Barrier::new(5));
                let threads: Vec<_> = (0..4)
                    .map(|thread| {
                        let phase = Arc::clone(&phase);
                        std::thread::spawn(move || {
                            let native: unsafe extern "C" fn(i64, i64, i64) -> i64 =
                                unsafe { std::mem::transmute(address) };
                            let mut oracle = NestedOracle::default();
                            let mut wrong = 0;
                            phase.wait();
                            for i in 0..512 {
                                let (outer, inner, salt) = concurrent_arguments(thread, i);
                                wrong += usize::from(
                                    unsafe { native(outer, inner, salt) }
                                        != oracle.observe(outer, inner, salt),
                                );
                            }
                            phase.wait();
                            phase.wait();
                            for i in 512..4096 {
                                let (outer, inner, salt) = concurrent_arguments(thread, i);
                                wrong += usize::from(
                                    unsafe { native(outer, inner, salt) }
                                        != oracle.observe(outer, inner, salt),
                                );
                            }
                            wrong
                        })
                    })
                    .collect();
                let mut paused_oracle = NestedOracle::default();
                let mut final_oracle = NestedOracle::default();
                for thread in 0..4 {
                    for i in 0..4096 {
                        let (outer, inner, salt) = concurrent_arguments(thread, i);
                        final_oracle.observe(outer, inner, salt);
                        if i < 512 {
                            paused_oracle.observe(outer, inner, salt);
                        }
                    }
                }
                phase.wait();
                phase.wait();
                let paused = training.branch_profile();
                // Release and join every worker before assertions so failures
                // cannot strand workers or dispose code while it is executing.
                phase.wait();
                let during = training.branch_profile();
                let results: Vec<_> = threads.into_iter().map(|thread| thread.join()).collect();
                let paused = paused.unwrap();
                let during = during.unwrap();
                paused_oracle.check(&paused, "nested_work", &context);
                for ((site, (lo_true, lo_false)), (hi_true, hi_false)) in during
                    .sites()
                    .iter()
                    .zip(paused_oracle.expected())
                    .zip(final_oracle.expected())
                {
                    assert!(
                        (lo_true..=hi_true).contains(&site.true_count)
                            && (lo_false..=hi_false).contains(&site.false_count),
                        "{context}: bounded concurrent {} snapshot",
                        site.block
                    );
                }
                for result in results {
                    assert_eq!(
                        result.unwrap(),
                        0,
                        "{context}: every native training result"
                    );
                }
                let final_profile = training.branch_profile().unwrap();
                final_oracle.check(&final_profile, "nested_work", &context);
                paused_oracle.check(&paused, "nested_work", &context);
                assert_eq!(empty.total_observations(), 0);
                assert!(final_profile.total_observations() > paused.total_observations());
                let final_jit =
                    CpuJit::compile_with_profile(CONCURRENT_NESTED_SOURCE, options, &final_profile)
                        .unwrap();
                assert_eq!(
                    final_jit.outer_unroll_annotations(),
                    usize::from(!final_unroll_outer_loops)
                );
                assert!(final_jit.branch_profile().is_err());
                let final_address = native_nested_address(&final_jit);
                let threads: Vec<_> = (0..4)
                    .map(|thread| {
                        std::thread::spawn(move || {
                            let native: unsafe extern "C" fn(i64, i64, i64) -> i64 =
                                unsafe { std::mem::transmute(final_address) };
                            let mut oracle = NestedOracle::default();
                            (0..4096)
                                .filter(|&i| {
                                    let (outer, inner, salt) = concurrent_arguments(thread, i);
                                    (unsafe { native(outer, inner, salt) })
                                        != oracle.observe(outer, inner, salt)
                                })
                                .count()
                        })
                    })
                    .collect();
                let results: Vec<_> = threads.into_iter().map(|thread| thread.join()).collect();
                for result in results {
                    assert_eq!(
                        result.unwrap(),
                        0,
                        "{context}: every final native nested result"
                    );
                }
                final_oracle.check(&training.branch_profile().unwrap(), "nested_work", &context);
                drop(training);
                final_oracle.check(&final_profile, "nested_work", &context);
                paused_oracle.check(&paused, "nested_work", &context);
                assert_eq!(empty.total_observations(), 0);
            }
        }
    }
}

const NESTED_IEEE_SOURCE: &str = r#"
@unsafe
fn nested64(output: GlobalMemory<F64>, input: GlobalMemory<F64>, a: F64, b: F64,
            outer: I64, inner: I64) -> F64 {
    let mut i: I64 = 0;
    let mut index: I64 = 0;
    let mut value: F64 = a;
    while i < outer {
        let mut j: I64 = 0;
        while j < inner {
            let product: F64 = value * b;
            value = product - 1.0;
            output[index] = product;
            output[index + 1] = value;
            index = index + 2;
            j = j + 1;
        }
        i = i + 1;
    }
    output[24] = input[0];
    output[25] = input[1];
    return value;
}
@unsafe
fn nested32(output: GlobalMemory<F32>, input: GlobalMemory<F32>, a: F32, b: F32,
            outer: I64, inner: I64) -> F32 {
    let mut i: I64 = 0;
    let mut index: I64 = 0;
    let mut value: F32 = a;
    while i < outer {
        let mut j: I64 = 0;
        while j < inner {
            let product: F32 = value * b;
            value = product - 1.0;
            output[index] = product;
            output[index + 1] = value;
            index = index + 2;
            j = j + 1;
        }
        i = i + 1;
    }
    output[24] = input[0];
    output[25] = input[1];
    return value;
}
"#;

// Operation barriers force separately rounded host results, so the oracle
// cannot fuse a multiply and subtraction while LLVM optimizes the nested loop.
#[inline(never)]
fn rounded_product64(a: f64, b: f64) -> f64 {
    std::hint::black_box(a * b)
}
#[inline(never)]
fn rounded_difference64(a: f64) -> f64 {
    std::hint::black_box(a - 1.0)
}
#[inline(never)]
fn rounded_product32(a: f32, b: f32) -> f32 {
    std::hint::black_box(a * b)
}
#[inline(never)]
fn rounded_difference32(a: f32) -> f32 {
    std::hint::black_box(a - 1.0)
}

fn compare_ieee64(actual: f64, expected: f64, context: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{context}: arithmetic NaN classification");
    } else {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{context}: exact rounded F64 bits"
        );
    }
}
fn compare_ieee32(actual: f32, expected: f32, context: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{context}: arithmetic NaN classification");
    } else {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{context}: exact rounded F32 bits"
        );
    }
}

fn exercise_nested_ieee(jit: &CpuJit, context: &str) -> NestedOracle {
    let pairs64 = [
        (1.0 + 2_f64.powi(-52), 1.0 - 2_f64.powi(-52)),
        (-0.0, 1.0),
        (0.0, -1.0),
        (f64::from_bits(1), 0.5),
        (f64::MIN_POSITIVE, 0.5),
        (f64::MAX, 2.0),
        (f64::INFINITY, 1.0),
        (f64::NEG_INFINITY, 1.0),
        (f64::INFINITY, 0.0),
        (f64::from_bits(0x7ff8_0000_1234_5678), 1.0),
    ];
    let pairs32 = [
        (1.0 + 2_f32.powi(-23), 1.0 - 2_f32.powi(-23)),
        (-0.0, 1.0),
        (0.0, -1.0),
        (f32::from_bits(1), 0.5),
        (f32::MIN_POSITIVE, 0.5),
        (f32::MAX, 2.0),
        (f32::INFINITY, 1.0),
        (f32::NEG_INFINITY, 1.0),
        (f32::INFINITY, 0.0),
        (f32::from_bits(0x7fc1_2345), 1.0),
    ];
    let bounds = [(-1, 3), (0, 0), (2, 0), (1, 1), (2, 3), (3, 1)];
    let mut oracle = NestedOracle::default();
    for &(outer, inner) in &bounds {
        for (pair64, pair32) in pairs64.iter().zip(pairs32) {
            oracle.observe(outer, inner, 0);
            let mut actual64 = [1437.25_f64; 30];
            let mut expected64 = actual64;
            let mut input64 = [
                f64::from_bits(0x7ff8_0000_1234_5678),
                f64::from_bits(0xfff0_0000_1234_5678),
            ];
            let mut actual32 = [1437.25_f32; 30];
            let mut expected32 = actual32;
            let mut input32 = [f32::from_bits(0x7fc1_2345), f32::from_bits(0xff81_2345)];
            let (mut value64, mut value32) = (pair64.0, pair32.0);
            let mut index = 1;
            for _ in 0..outer.max(0) {
                for _ in 0..inner.max(0) {
                    let product64 = rounded_product64(value64, pair64.1);
                    value64 = rounded_difference64(product64);
                    expected64[index] = product64;
                    expected64[index + 1] = value64;
                    let product32 = rounded_product32(value32, pair32.1);
                    value32 = rounded_difference32(product32);
                    expected32[index] = product32;
                    expected32[index + 1] = value32;
                    index += 2;
                }
            }
            expected64[25] = input64[0];
            expected64[26] = input64[1];
            expected32[25] = input32[0];
            expected32[26] = input32[1];
            let returned64 = call(
                jit,
                "nested64",
                &[
                    V::Pointer(unsafe { actual64.as_mut_ptr().add(1) }.cast()),
                    V::Pointer(input64.as_mut_ptr().cast()),
                    V::F64(pair64.0),
                    V::F64(pair64.1),
                    V::I64(outer),
                    V::I64(inner),
                ],
                context,
            );
            let returned32 = call(
                jit,
                "nested32",
                &[
                    V::Pointer(unsafe { actual32.as_mut_ptr().add(1) }.cast()),
                    V::Pointer(input32.as_mut_ptr().cast()),
                    V::F32(pair32.0),
                    V::F32(pair32.1),
                    V::I64(outer),
                    V::I64(inner),
                ],
                context,
            );
            let V::F64(returned64) = returned64 else {
                panic!("{context}: F64 ABI");
            };
            let V::F32(returned32) = returned32 else {
                panic!("{context}: F32 ABI");
            };
            compare_ieee64(returned64, value64, context);
            compare_ieee32(returned32, value32, context);
            for index in 0..30 {
                if index == 25 || index == 26 {
                    // Pass-through memory preserves the exact quiet/signaling
                    // NaN payload; arithmetic NaN payload selection is unrestricted.
                    assert_eq!(
                        actual64[index].to_bits(),
                        expected64[index].to_bits(),
                        "{context}: copied F64 payload"
                    );
                    assert_eq!(
                        actual32[index].to_bits(),
                        expected32[index].to_bits(),
                        "{context}: copied F32 payload"
                    );
                } else {
                    compare_ieee64(actual64[index], expected64[index], context);
                    compare_ieee32(actual32[index], expected32[index], context);
                }
            }
        }
    }
    oracle
}

fn check_nested_ieee_profile(profile: &BranchProfile, oracle: NestedOracle, context: &str) {
    assert_eq!(profile.sites().len(), 4);
    for function in ["nested64", "nested32"] {
        let sites: Vec<_> = profile
            .sites()
            .iter()
            .filter(|site| site.function == function)
            .collect();
        assert_eq!(sites.len(), 2);
        for (site, expected) in sites.iter().zip(oracle.expected()) {
            assert_eq!(
                (site.true_count, site.false_count),
                expected,
                "{context}: exact IEEE loop {}",
                site.block
            );
        }
    }
}

#[test]
fn outer_policy_preserves_ieee_operations_inside_eligible_nested_loops() {
    for opt_level in 0..=3 {
        for codegen_opt_level in [None, Some(1)] {
            let defaults = JitOptions {
                opt_level,
                codegen_opt_level,
                training_opt_level: Some(1),
                profile_loop_controls: true,
                profile_loop_edge_counters: true,
                ..JitOptions::default()
            };
            let outer_disabled = JitOptions {
                final_unroll_outer_loops: false,
                ..defaults
            };
            let context = format!("nested IEEE finalIR O{opt_level}, native {codegen_opt_level:?}");
            let training_default =
                CpuJit::compile_instrumented(NESTED_IEEE_SOURCE, defaults).unwrap();
            let training_disabled =
                CpuJit::compile_instrumented(NESTED_IEEE_SOURCE, outer_disabled).unwrap();
            assert_eq!(
                training_default.optimized_ir(),
                training_disabled.optimized_ir()
            );
            assert_eq!(training_default.outer_unroll_annotations(), 0);
            assert_eq!(training_disabled.outer_unroll_annotations(), 0);
            let oracle = exercise_nested_ieee(&training_default, &context);
            exercise_nested_ieee(&training_disabled, &context);
            let profile = training_default.branch_profile().unwrap();
            assert_eq!(profile, training_disabled.branch_profile().unwrap());
            check_nested_ieee_profile(&profile, oracle, &context);
            let variants = [
                CpuJit::compile_with_options(NESTED_IEEE_SOURCE, defaults).unwrap(),
                CpuJit::compile_with_options(NESTED_IEEE_SOURCE, outer_disabled).unwrap(),
                CpuJit::compile_with_profile(NESTED_IEEE_SOURCE, defaults, &profile).unwrap(),
                CpuJit::compile_with_profile(NESTED_IEEE_SOURCE, outer_disabled, &profile).unwrap(),
            ];
            for (index, jit) in variants.iter().enumerate() {
                assert_eq!(
                    jit.outer_unroll_annotations(),
                    if index % 2 == 0 { 0 } else { 2 }
                );
                exercise_nested_ieee(jit, &context);
            }
            assert_eq!(training_default.branch_profile().unwrap(), profile);
            assert_eq!(training_disabled.branch_profile().unwrap(), profile);
        }
    }
}
