//! A cheaper instrumented IR tier must preserve source outcomes and exact
//! original-site observations before compiling unchanged profiled O3 code.
//! These bounded oracles inspect complete host/object contents; they are not
//! a proof for arbitrary programs or a training-tier performance benchmark.
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
fn instrumented_tiers_preserve_exact_profiles_and_final_o3_full_results() {
    assert_eq!(JitOptions::default().training_opt_level, None);
    assert!(JitOptions::default().verify_each_pass);
    for codegen_opt_level in [None, Some(1)] {
        let inherited = JitOptions {
            opt_level: 3,
            codegen_opt_level,
            ..JitOptions::default()
        };
        let ordinary = CpuJit::compile_with_options(SOURCE, inherited).unwrap();
        exercise(&ordinary, &UNSEEN, "baseline ordinary");
        let baseline_training = CpuJit::compile_instrumented(SOURCE, inherited).unwrap();
        let oracle = exercise(&baseline_training, &TRAINING, "baseline training");
        let profile = baseline_training.branch_profile().unwrap();
        oracle.check(&profile, "baseline training");
        assert!(profile.total_observations() > 0);
        assert_eq!(baseline_training.compile_timings().verification_checks, 2);
        let profiled = CpuJit::compile_with_profile(SOURCE, inherited, &profile).unwrap();
        assert!(profiled.profiled_branches() > 0);
        assert_eq!(profiled.compile_timings().verification_checks, 3);
        exercise(&profiled, &UNSEEN, "baseline profiled unseen");
        assert!(!profiled.optimized_ir().contains("atomicrmw"));

        for tier in 0..=3 {
            let options = JitOptions {
                training_opt_level: Some(tier),
                ..inherited
            };
            let context =
                format!("training IR O{tier}, codegen={codegen_opt_level:?}, final IR O3");
            let ordinary_override = CpuJit::compile_with_options(SOURCE, options).unwrap();
            assert_eq!(
                ordinary_override.optimized_ir(),
                ordinary.optimized_ir(),
                "{context}: training override is ignored by ordinary compilation"
            );
            exercise(&ordinary_override, &UNSEEN, &context);
            let training = CpuJit::compile_instrumented(SOURCE, options).unwrap();
            assert_eq!(
                training.branch_profile().unwrap().total_observations(),
                0,
                "{context}: compilation does not execute source"
            );
            assert_eq!(training.compile_timings().verification_checks, 2);
            if tier == 3 {
                assert_eq!(
                    training.optimized_ir(),
                    baseline_training.optimized_ir(),
                    "{context}: explicit O3 training equals inherited training"
                );
            }
            let oracle = exercise(&training, &TRAINING, &context);
            let measured = training.branch_profile().unwrap();
            oracle.check(&measured, &context);
            assert_eq!(
                measured, profile,
                "{context}: all original counts/sites/fingerprint"
            );
            let final_jit = CpuJit::compile_with_profile(SOURCE, options, &measured).unwrap();
            assert_eq!(
                final_jit.optimized_ir(),
                profiled.optimized_ir(),
                "{context}: final O3 IR ignores the training-only override"
            );
            assert_eq!(final_jit.profiled_branches(), profiled.profiled_branches());
            assert_eq!(
                final_jit.profile_selection_optimization(),
                profiled.profile_selection_optimization()
            );
            assert_eq!(final_jit.compile_timings().verification_checks, 3);
            exercise(&final_jit, &UNSEEN, &context);
            assert_eq!(
                training.branch_profile().unwrap(),
                measured,
                "{context}: final compilation/calls do not execute the trainer"
            );
            drop(final_jit);
            let observed_before = training.branch_profile().unwrap().total_observations();
            exercise(&training, &UNSEEN, &context);
            assert!(
                training.branch_profile().unwrap().total_observations() > observed_before,
                "{context}: trainer remains usable after final session disposal"
            );
            assert_eq!(measured, profile, "{context}: snapshots own their counts");
        }
    }
}

#[test]
fn invalid_training_tier_is_rejected_in_every_compilation_mode() {
    let source = "fn choose(x: I64) -> I64 { if x < 0 { return -1; } return 2; }";
    let training = CpuJit::compile_instrumented(source, JitOptions::default()).unwrap();
    let profile = training.branch_profile().unwrap();
    let invalid = JitOptions {
        training_opt_level: Some(4),
        ..JitOptions::default()
    };
    for (mode, result) in [
        ("ordinary", CpuJit::compile_with_options(source, invalid)),
        (
            "instrumented",
            CpuJit::compile_instrumented(source, invalid),
        ),
        (
            "profile-use",
            CpuJit::compile_with_profile(source, invalid, &profile),
        ),
    ] {
        match result {
            Ok(_) => panic!("invalid training_opt_level accepted in {mode}"),
            Err(error) => assert!(
                error.to_string().contains("training_opt_level"),
                "{mode}: {error}"
            ),
        }
    }
}
