//! Execute deterministic valid programs against independent intent-level oracles.
//! These tests compare complete native values/memory, not process exit codes or
//! agreement between two backends sharing the same LLVM emitter. They do not
//! establish equivalence for arbitrary Y programs: generated scalar templates
//! and bounded memory/runtime scenarios deliberately avoid undefined operations.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::ffi::c_void;
use y::cpu_jit::{CpuJit, JitOptions, JitValue as V};

const ROOT_SEED: u64 = 0x596a_6974_6469_6666;
const VARIANTS: usize = 16;
const INPUTS: usize = 128;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // SplitMix64 is only input generation; it is not the function under test.
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

struct Variant {
    salt: u64,
    shift: u32,
    increment: u64,
    divisor: i64,
    source: String,
}

fn variants() -> Vec<Variant> {
    let mut rng = Rng(ROOT_SEED);
    (0..VARIANTS)
        .map(|index| {
            // Source integer literals fit I64. Full-width values enter as ABI
            // parameters, so high-bit coverage does not depend on literal parsing.
            let salt = rng.next() & i64::MAX as u64;
            let shift = (rng.next() % 63 + 1) as u32;
            let increment = rng.next() & i64::MAX as u64;
            let divisor = (rng.next() % 31 + 2) as i64;
            let complement = 64 - shift;
            let source = format!(
                r#"
@unsafe
fn rotate_{index}(input: U64, rounds: I64) -> U64 {{
    let mut value: U64 = input ^ {salt};
    let mut i: I64 = 0;
    while i < rounds {{
        if (value & 1) == 0 {{ value = (value << {shift}) | (value >> {complement}); }}
        else {{ value = (value >> {shift}) | (value << {complement}); }}
        value += {increment};
        i = i + 1;
    }}
    return value;
}}
fn quotient_{index}(input: I64) -> I64 {{ return input / {divisor}; }}
fn restore_{index}(input: I64) -> I64 {{
    let q: I64 = input / {divisor};
    let r: I64 = input % {divisor};
    let rebuilt: I64 = recompose(q, r, {divisor});
    if r < 0 {{ return rebuilt ^ {salt}; }}
    return rebuilt ^ {half_salt};
}}
"#,
                half_salt = salt >> 1,
            );
            Variant {
                salt,
                shift,
                increment,
                divisor,
                source,
            }
        })
        .collect()
}

const FIXED_SOURCE: &str = r#"
fn recompose(q: I64, r: I64, d: I64) -> I64 { return q * d + r; }
fn scalar_value(payload: I64, index: I64) -> I64 { return payload ^ index; }
fn mixed(a: U8, b: I16, c: U32, d: I64) -> I64 { return a / b + c / d; }
fn equal_width_order(a: I32, b: U32) -> bool { return a > b; }
fn signed_shift(a: I32, amount: U64) -> I64 { return a >> amount; }
fn mixed_abi(a: I64, b: F64, c: I8, d: F32, e: U64, f: bool, g: I32, h: I64, i: U16) -> I64 {
    if f && b == 1.25 && d == 2.5 && (e >> 63) == 1 { return a + c + g + h + i; }
    return -123456789;
}
@unsafe
fn memory(p: GlobalMemory<I64>, q: GlobalMemory<I64>, mask: I64, n: I64) -> I64 {
    let mut i: I64 = 0;
    while i < n {
        let old: I64 = p[i];
        p[i] = q[7 - i] ^ mask;
        q[7 - i] = old;
        i = i + 1;
    }
    return p[0] ^ p[7];
}
fn alter_copy(values: [I64; 8], replacement: I64) -> I64 {
    values[0] = replacement;
    return values[0] ^ values[7];
}
@unsafe
fn arrays(output: GlobalMemory<I64>, input: I64) {
    let mut original: [I64; 8] = {};
    for i in 0..8 { original[i] = input ^ i; }
    let mut copy: [I64; 8] = original;
    original[0] = 12345;
    let result: I64 = alter_copy(copy, 67890);
    for i in 0..8 { output[i] = copy[i]; }
    output[8] = original[0];
    output[9] = result;
}
@unsafe
fn record(output: GlobalMemory<I64>, digit: I64, result: bool) -> bool {
    output[0] = output[0] + 1;
    output[1] = output[1] * 10 + digit;
    return result;
}
@unsafe
fn short_circuit(output: GlobalMemory<I64>, a: bool, b: bool, c: bool, d: bool) -> bool {
    let result: bool = (record(output, 1, a) || record(output, 2, b)) &&
                       (record(output, 3, c) || record(output, 4, d));
    output[2] = result;
    return result;
}
@unsafe
fn owned(output: GlobalMemory<I64>, payload: I64, n: I64, element_size: I64, token: char) {
    let mut text: String = "";
    let mut suffix: String = "xy";
    let mut words: Vec = Vec_new(element_size);
    let mut i: I64 = 0;
    while i < n {
        let value: I64 = scalar_value(payload, i);
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
        output[70 + i] = ychar_to_ascii(String_char_at(&text, i));
        i = i + 1;
    }
    output[230] = Vec_len(&words);
    output[231] = String_len(&text);
    output[232] = ychar_to_ascii(String_char_at(&text, -1));
    output[233] = ychar_to_ascii(String_char_at(&text, String_len(&text)));
    Vec_free(&mut words);
    String_free(&mut text);
    String_free(&mut suffix);
    output[234] = Vec_len(&words) + String_len(&text) + String_len(&suffix);
    output[235] = ychar_to_ascii(String_char_at(&text, 0));
}
fn exact64(a: F64, b: F64) -> F64 { return a * 0.5 + b; }
fn exact32(a: F32, b: F32) -> F32 { return a * 0.5 + b; }
fn rounded64(a: F64, b: F64) -> F64 { let product: F64 = a * b; return product - 1.0; }
fn rounded32(a: F32, b: F32) -> F32 { let product: F32 = a * b; return product - 1.0; }
fn precise_literal() -> F64 { return 1.0000001; }
fn ieee64(a: F64) -> F64 { return a; }
fn ieee32(a: F32) -> F32 { return a; }
fn compare64(a: F64, b: F64) -> I32 {
    let mut flags: I32 = 0;
    if a == b { flags += 1; }
    if a != b { flags += 2; }
    if a < b { flags += 4; }
    if a >= b { flags += 8; }
    return flags;
}
"#;

fn options(
    level: u8,
    enabled: bool,
    verify_each_pass: bool,
    codegen_opt_level: Option<u8>,
) -> JitOptions {
    JitOptions {
        opt_level: level,
        recognize_rotates: enabled,
        optimize_runtime: enabled,
        optimize_runtime_mutations: enabled,
        optimize_runtime_copies: enabled,
        optimize_helper_effects: enabled,
        optimize_call_adapters: enabled,
        verify_each_pass,
        codegen_opt_level,
        ..JitOptions::default()
    }
}

fn call(jit: &CpuJit, name: &str, args: &[V], context: &str, source: &str) -> V {
    // All pointer-free inputs have exactly the function's checked ABI tags.
    unsafe { jit.call(name, args) }
        .unwrap_or_else(|error| panic!("{context}, {name}, args={args:?}: {error}\n{source}"))
}

fn signed_input(case: usize, random: u64) -> i64 {
    const EDGES: [i64; 12] = [
        i64::MIN,
        i64::MIN + 1,
        i64::MAX,
        i64::MAX - 1,
        -4294967297,
        4294967297,
        -1,
        0,
        1,
        -32,
        32,
        0x123456789abcdef,
    ];
    if case < EDGES.len() {
        EDGES[case]
    } else {
        random as i64
    }
}

fn unsigned_input(case: usize, random: u64) -> u64 {
    const EDGES: [u64; 10] = [
        0,
        1,
        u64::MAX,
        1 << 63,
        (1 << 63) + 1,
        (1 << 63) - 1,
        u32::MAX as u64,
        1 << 32,
        0xaaaaaaaaaaaaaaaa,
        0x5555555555555555,
    ];
    if case < EDGES.len() {
        EDGES[case]
    } else {
        random
    }
}

#[test]
fn seeded_valid_programs_match_full_results_with_and_without_optimizations() {
    let variants = variants();
    let source = format!(
        "{}{}",
        FIXED_SOURCE,
        variants
            .iter()
            .map(|v| v.source.as_str())
            .collect::<String>()
    );
    let configurations = [(0, true), (0, false), (3, true), (3, false)]
        .into_iter()
        .flat_map(|(level, verify_each_pass)| {
            [None, Some(2)].map(move |codegen| (level, verify_each_pass, codegen))
        });
    for (level, verify_each_pass, codegen) in configurations {
        for enabled in [false, true] {
            let config = format!("IR O{level}, codegen={codegen:?}, transforms={enabled}, verify_each_pass={verify_each_pass}, root_seed={ROOT_SEED:#018x}");
            let jit = CpuJit::compile_with_options(
                &source,
                options(level, enabled, verify_each_pass, codegen),
            )
            .unwrap_or_else(|error| panic!("{config}: {error}\n{source}"));
            let mut rng = Rng(ROOT_SEED ^ 0x0123456789abcdef);
            for case in 0..INPUTS {
                let seed = rng.next();
                let context = format!("{config}, case={case}, seed={seed:#018x}");
                let signed = signed_input(case, seed);
                let unsigned = unsigned_input(case, seed);
                for (index, variant) in variants.iter().enumerate() {
                    let rounds = (seed >> (index % 16)) % 9;
                    let mut expected = unsigned ^ variant.salt;
                    for _ in 0..rounds {
                        expected = if expected % 2 == 0 {
                            expected.rotate_left(variant.shift)
                        } else {
                            expected.rotate_right(variant.shift)
                        };
                        expected = expected.wrapping_add(variant.increment);
                    }
                    let name = format!("rotate_{index}");
                    assert_eq!(
                        call(
                            &jit,
                            &name,
                            &[V::U64(unsigned), V::I64(rounds as i64)],
                            &context,
                            &variant.source
                        ),
                        V::U64(expected),
                        "{context}, {name}\n{}",
                        variant.source
                    );
                    // Evaluate the division in a wider mathematical domain. The
                    // emitted q*d+r is bounded by input, including I64::MIN.
                    let quotient = i128::from(signed) / i128::from(variant.divisor);
                    let remainder = i128::from(signed) % i128::from(variant.divisor);
                    let name = format!("quotient_{index}");
                    assert_eq!(
                        call(&jit, &name, &[V::I64(signed)], &context, &variant.source),
                        V::I64(quotient as i64),
                        "{context}, {name}\n{}",
                        variant.source
                    );
                    let name = format!("restore_{index}");
                    let mask = if remainder < 0 {
                        variant.salt
                    } else {
                        variant.salt >> 1
                    };
                    assert_eq!(
                        call(&jit, &name, &[V::I64(signed)], &context, &variant.source),
                        V::I64(signed ^ mask as i64),
                        "{context}, {name}\n{}",
                        variant.source
                    );
                }
                check_mixed(&jit, case, seed, &context);
                check_memory(&jit, case, seed, &context);
                check_runtime(&jit, case, seed, &context);
                check_floats(&jit, case, seed, &context);
            }
        }
    }
}

fn check_mixed(jit: &CpuJit, case: usize, seed: u64, context: &str) {
    let byte = seed as u8;
    let denominator = [-32768_i16, -31, -2, 1, 3, 32767][case % 6];
    let word = (seed >> 16) as u32;
    let wide_denominator = [-i64::MAX, -17, -1, 1, 31, i64::MAX][case % 6];
    let expected = i128::from(byte) / i128::from(denominator)
        + i128::from(word) / i128::from(wide_denominator);
    assert_eq!(
        call(
            jit,
            "mixed",
            &[
                V::U8(byte),
                V::I16(denominator),
                V::U32(word),
                V::I64(wide_denominator)
            ],
            context,
            FIXED_SOURCE
        ),
        V::I64(expected as i64),
        "{context}: mixed-width divisions\n{FIXED_SOURCE}"
    );
    let signed_word = seed as i32;
    assert_eq!(
        call(
            jit,
            "equal_width_order",
            &[V::I32(signed_word), V::U32(word)],
            context,
            FIXED_SOURCE
        ),
        V::Bool((signed_word as u32) > word),
        "{context}: equal-width unsigned ordering\n{FIXED_SOURCE}"
    );
    let amount = (seed >> 32) % 32;
    assert_eq!(
        call(
            jit,
            "signed_shift",
            &[V::I32(signed_word), V::U64(amount)],
            context,
            FIXED_SOURCE
        ),
        V::I64(i64::from(signed_word >> amount)),
        "{context}: signed shift with unsigned count\n{FIXED_SOURCE}"
    );
    let a = (seed & 4095) as i64 - 2048;
    let c = (seed >> 8) as i8;
    let g = (seed >> 32) as i32;
    let h = (seed >> 48) as i64;
    let i = (seed >> 16) as u16;
    let accept = case % 3 != 0;
    let high = seed | (1 << 63);
    let expected = if accept {
        (i128::from(a) + i128::from(c) + i128::from(g) + i128::from(h) + i128::from(i)) as i64
    } else {
        -123456789
    };
    assert_eq!(
        call(
            jit,
            "mixed_abi",
            &[
                V::I64(a),
                V::F64(1.25),
                V::I8(c),
                V::F32(2.5),
                V::U64(high),
                V::Bool(accept),
                V::I32(g),
                V::I64(h),
                V::U16(i)
            ],
            context,
            FIXED_SOURCE
        ),
        V::I64(expected),
        "{context}: nine mixed ABI arguments\n{FIXED_SOURCE}"
    );
}

fn check_memory(jit: &CpuJit, case: usize, seed: u64, context: &str) {
    let memory: unsafe extern "C" fn(*mut i64, *mut i64, i64, i64) -> i64 =
        unsafe { std::mem::transmute(jit.function_address("memory").unwrap()) };
    let n = case % 9;
    for second_base in [1_usize, 3, 9] {
        let mut rng = Rng(seed);
        let mut actual = std::array::from_fn::<_, 18, _>(|_| rng.next() as i64);
        let mut expected = actual;
        for index in 0..n {
            let left = 1 + index;
            let right = second_base + 7 - index;
            let saved = expected[left];
            expected[left] = expected[right] ^ seed as i64;
            expected[right] = saved;
        }
        let result = unsafe {
            memory(
                actual.as_mut_ptr().add(1),
                actual.as_mut_ptr().add(second_base),
                seed as i64,
                n as i64,
            )
        };
        assert_eq!(
            actual, expected,
            "{context}: all memory, base={second_base}, n={n}\n{FIXED_SOURCE}"
        );
        assert_eq!(
            result,
            expected[1] ^ expected[8],
            "{context}: memory return\n{FIXED_SOURCE}"
        );
    }

    let mut actual = [0x123456789abcdef_i64; 12];
    let mut expected = actual;
    for index in 0..8 {
        expected[index + 1] = seed as i64 ^ index as i64;
    }
    expected[9] = 12345;
    expected[10] = 67890 ^ (seed as i64 ^ 7);
    let output = V::Pointer(unsafe { actual.as_mut_ptr().add(1) }.cast::<c_void>());
    assert_eq!(
        call(
            jit,
            "arrays",
            &[output, V::I64(seed as i64)],
            context,
            FIXED_SOURCE
        ),
        V::Void
    );
    assert_eq!(
        actual, expected,
        "{context}: local/by-value arrays and memory guards\n{FIXED_SOURCE}"
    );

    let bits = case % 16;
    let mut actual = [0_i64, 0, -1, 0x123456789abcdef];
    let mut expected = actual;
    // State-machine oracle records only the steps reachable from the four
    // supplied truth values, rather than evaluating the emitted expression.
    let mut visited = vec![1_i64];
    let left = if bits & 1 != 0 {
        true
    } else {
        visited.push(2);
        bits & 2 != 0
    };
    let result = if !left {
        false
    } else {
        visited.push(3);
        if bits & 4 != 0 {
            true
        } else {
            visited.push(4);
            bits & 8 != 0
        }
    };
    expected[0] = visited.len() as i64;
    expected[1] = visited.iter().fold(0, |trace, digit| trace * 10 + digit);
    expected[2] = i64::from(result);
    let args = [
        V::Pointer(actual.as_mut_ptr().cast::<c_void>()),
        V::Bool(bits & 1 != 0),
        V::Bool(bits & 2 != 0),
        V::Bool(bits & 4 != 0),
        V::Bool(bits & 8 != 0),
    ];
    assert_eq!(
        call(jit, "short_circuit", &args, context, FIXED_SOURCE),
        V::Bool(result),
        "{context}: short circuit return\n{FIXED_SOURCE}"
    );
    assert_eq!(
        actual, expected,
        "{context}: complete short circuit side effects\n{FIXED_SOURCE}"
    );
}

fn check_runtime(jit: &CpuJit, case: usize, seed: u64, context: &str) {
    let n = [0_usize, 1, 7, 8, 9, 15, 16, 31, 32, 63, 64, 65][case % 12];
    let token = [0_u8, 65, 200, 255][case % 4];
    let mut actual = [0x123456789abcdef_i64; 258];
    let mut expected = actual;
    for index in 0..n {
        expected[1 + index] = seed as i64 ^ index as i64;
    }
    let mut text: Vec<u8> = (0..n)
        .map(|index| if index % 2 == 0 { token } else { b'Z' })
        .collect();
    text.extend_from_slice(b"xy");
    text.extend_from_within(..);
    for (index, byte) in text.iter().enumerate() {
        expected[1 + 70 + index] = i64::from(*byte);
    }
    expected[231] = n as i64;
    expected[232] = text.len() as i64;
    expected[233..237].fill(0);
    let output = V::Pointer(unsafe { actual.as_mut_ptr().add(1) }.cast::<c_void>());
    assert_eq!(
        call(
            jit,
            "owned",
            &[
                output,
                V::I64(seed as i64),
                V::I64(n as i64),
                V::I64(8),
                V::U8(token)
            ],
            context,
            FIXED_SOURCE
        ),
        V::Void
    );
    assert_eq!(
        actual, expected,
        "{context}: complete String/Vec contents, n={n}, token={token}\n{FIXED_SOURCE}"
    );
}

fn check_floats(jit: &CpuJit, case: usize, seed: u64, context: &str) {
    // Values on a small binary grid make every arithmetic step exact in F32
    // and F64. Expected results come from the integer grid, without reproducing
    // the floating expression or allowing tolerance to hide bit differences.
    let a = (seed & 65535) as i64 - 32768;
    let b = ((seed >> 16) & 65535) as i64 - 32768;
    let expected = (a + 2 * b) as f64 / 16.0;
    let actual = call(
        jit,
        "exact64",
        &[V::F64(a as f64 / 8.0), V::F64(b as f64 / 8.0)],
        context,
        FIXED_SOURCE,
    );
    assert_eq!(
        actual.bits(),
        expected.to_bits(),
        "{context}: exact F64\n{FIXED_SOURCE}"
    );
    let actual = call(
        jit,
        "exact32",
        &[V::F32(a as f32 / 8.0), V::F32(b as f32 / 8.0)],
        context,
        FIXED_SOURCE,
    );
    assert_eq!(
        actual.bits(),
        u64::from((expected as f32).to_bits()),
        "{context}: exact F32\n{FIXED_SOURCE}"
    );
    // (1+2^-27)*(1-2^-27) = 1-2^-54. Rounding the F64
    // product first gives 1 (the even tie), then subtracting 1 gives +0.
    // A fused/reassociated operation would instead retain the negative term.
    let actual = call(
        jit,
        "rounded64",
        &[
            V::F64(f64::from_bits(0x3ff0000002000000)),
            V::F64(f64::from_bits(0x3feffffffc000000)),
        ],
        context,
        FIXED_SOURCE,
    );
    assert_eq!(
        actual.bits(),
        0,
        "{context}: separate F64 rounding\n{FIXED_SOURCE}"
    );
    // The F32 analogue has exact product 1-2^-26, rounded to 1.
    let actual = call(
        jit,
        "rounded32",
        &[
            V::F32(f32::from_bits(0x3f800400)),
            V::F32(f32::from_bits(0x3f7ff800)),
        ],
        context,
        FIXED_SOURCE,
    );
    assert_eq!(
        actual.bits(),
        0,
        "{context}: separate F32 rounding\n{FIXED_SOURCE}"
    );
    assert_eq!(
        call(jit, "precise_literal", &[], context, FIXED_SOURCE).bits(),
        1.0000001_f64.to_bits(),
        "{context}: precise literal\n{FIXED_SOURCE}"
    );

    const IEEE64: [u64; 10] = [
        0,
        1 << 63,
        1,
        (1 << 63) | 1,
        0x3ff0000000000000,
        0xbff0000000000000,
        0x7ff0000000000000,
        0xfff0000000000000,
        0x7ff8000000001234,
        0xfff8000000004321,
    ];
    const IEEE32: [u32; 10] = [
        0,
        1 << 31,
        1,
        (1 << 31) | 1,
        0x3f800000,
        0xbf800000,
        0x7f800000,
        0xff800000,
        0x7fc01234,
        0xffc04321,
    ];
    let left = IEEE64[case % IEEE64.len()];
    let right = IEEE64[(case / IEEE64.len()) % IEEE64.len()];
    assert_eq!(
        call(
            jit,
            "ieee64",
            &[V::F64(f64::from_bits(left))],
            context,
            FIXED_SOURCE
        )
        .bits(),
        left,
        "{context}: F64 payload\n{FIXED_SOURCE}"
    );
    let single = IEEE32[case % IEEE32.len()];
    assert_eq!(
        call(
            jit,
            "ieee32",
            &[V::F32(f32::from_bits(single))],
            context,
            FIXED_SOURCE
        )
        .bits(),
        u64::from(single),
        "{context}: F32 payload\n{FIXED_SOURCE}"
    );
    let a = f64::from_bits(left);
    let b = f64::from_bits(right);
    let expected = if a.is_nan() || b.is_nan() {
        2
    } else if a == b {
        9
    } else if a < b {
        6
    } else {
        10
    };
    assert_eq!(
        call(
            jit,
            "compare64",
            &[V::F64(a), V::F64(b)],
            context,
            FIXED_SOURCE
        ),
        V::I32(expected),
        "{context}: IEEE ordering\n{FIXED_SOURCE}"
    );
}
