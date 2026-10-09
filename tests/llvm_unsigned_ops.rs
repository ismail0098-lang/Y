//! Native results across LLVM optimization levels, including inputs whose
//! high bits distinguish signed from unsigned instructions. Expectations are
//! constants/Rust arithmetic rather than two expressions from the Y emitter.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use y::cpu_jit::{CpuJit, JitOptions};

fn compile(source: &str, opt_level: u8) -> CpuJit {
    CpuJit::compile_with_options(
        source,
        JitOptions {
            opt_level,
            ..JitOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("O{opt_level}: {error}\n{source}"))
}

macro_rules! entry {
    ($jit:expr, $name:literal, $signature:ty) => {
        std::mem::transmute::<usize, $signature>($jit.function_address($name).unwrap())
    };
}

#[test]
fn unsigned_high_bits_choose_division_remainder_shift_and_ordering() {
    let source = r#"
fn quotient(a: U32, b: U32) -> U32 { return a / b; }
fn remainder(a: U32, b: U32) -> U32 { return a % b; }
fn shift(a: U32, b: U32) -> U32 { return a >> b; }
fn above(a: U32, b: U32) -> I32 { return a > b; }
fn below(a: U32, b: U32) -> I32 { return a < b; }
fn at_least(a: U32, b: U32) -> I32 { return a >= b; }
fn at_most(a: U32, b: U32) -> I32 { return a <= b; }
fn byte_div(a: U8, b: U8) -> U8 { return a / b; }
fn word_shift(a: U16, b: U16) -> U16 { return a >> b; }
fn signed_div(a: I32, b: I32) -> I32 { return a / b; }
fn signed_rem(a: I32, b: I32) -> I32 { return a % b; }
"#;
    for opt in [0, 3] {
        let jit = compile(source, opt);
        unsafe {
            let quotient = entry!(jit, "quotient", unsafe extern "C" fn(u32, u32) -> u32);
            let remainder = entry!(jit, "remainder", unsafe extern "C" fn(u32, u32) -> u32);
            let shift = entry!(jit, "shift", unsafe extern "C" fn(u32, u32) -> u32);
            let above = entry!(jit, "above", unsafe extern "C" fn(u32, u32) -> i32);
            let below = entry!(jit, "below", unsafe extern "C" fn(u32, u32) -> i32);
            let at_least = entry!(jit, "at_least", unsafe extern "C" fn(u32, u32) -> i32);
            let at_most = entry!(jit, "at_most", unsafe extern "C" fn(u32, u32) -> i32);
            let byte_div = entry!(jit, "byte_div", unsafe extern "C" fn(u8, u8) -> u8);
            let word_shift = entry!(jit, "word_shift", unsafe extern "C" fn(u16, u16) -> u16);
            let signed_div = entry!(jit, "signed_div", unsafe extern "C" fn(i32, i32) -> i32);
            let signed_rem = entry!(jit, "signed_rem", unsafe extern "C" fn(i32, i32) -> i32);
            assert_eq!(quotient(3_000_000_000, 7), 428_571_428);
            assert_eq!(remainder(3_000_000_000, 7), 4);
            assert_eq!(shift(0xf000_0000, 4), 0x0f00_0000);
            assert_eq!(above(3_000_000_000, 2), 1);
            assert_eq!(above(2, 3_000_000_000), 0);
            assert_eq!(below(2, 3_000_000_000), 1);
            assert_eq!(below(3_000_000_000, 2), 0);
            assert_eq!(at_least(3_000_000_000, 3_000_000_000), 1);
            assert_eq!(at_least(2, 3_000_000_000), 0);
            assert_eq!(at_most(3_000_000_000, 3_000_000_000), 1);
            assert_eq!(at_most(3_000_000_000, 2), 0);
            assert_eq!(byte_div(200, 3), 66);
            assert_eq!(word_shift(50_000, 2), 12_500);
            assert_eq!(signed_div(-9, 2), -4);
            assert_eq!(signed_rem(-9, 2), -1);
        }
    }
}

#[test]
fn unsigned_function_results_and_nested_expressions_keep_their_signedness() {
    let source = r#"
fn high() -> U64 { let one: U64 = 1; return (one << 63) + 5; }
fn quotient() -> U64 { return high() / 3; }
fn remainder() -> U64 { return high() % 3; }
fn shift() -> U64 { return high() >> 1; }
fn above_zero() -> I32 { return high() > 0; }
fn nested() -> U64 { return (high() >> 1) / 2; }
fn inferred() -> U64 {
    let one: U64 = 1;
    let value = (one << 63) + 5;
    return value >> 1;
}
"#;
    for opt in [0, 3] {
        let jit = compile(source, opt);
        unsafe {
            let high = entry!(jit, "high", unsafe extern "C" fn() -> u64);
            let quotient = entry!(jit, "quotient", unsafe extern "C" fn() -> u64);
            let remainder = entry!(jit, "remainder", unsafe extern "C" fn() -> u64);
            let shift = entry!(jit, "shift", unsafe extern "C" fn() -> u64);
            let above_zero = entry!(jit, "above_zero", unsafe extern "C" fn() -> i32);
            let nested = entry!(jit, "nested", unsafe extern "C" fn() -> u64);
            let inferred = entry!(jit, "inferred", unsafe extern "C" fn() -> u64);
            let expected = 0x8000_0000_0000_0005_u64;
            assert_eq!(high(), expected);
            assert_eq!(quotient(), expected / 3);
            assert_eq!(remainder(), expected % 3);
            assert_eq!(shift(), expected >> 1);
            assert_eq!(above_zero(), 1);
            assert_eq!(nested(), (expected >> 1) / 2);
            assert_eq!(inferred(), expected >> 1);
        }
    }
}

#[test]
fn mixed_widths_zero_extend_sources_and_preserve_wider_signed_arithmetic() {
    // Wider signed types represent narrower unsigned operands. At equal
    // widths unsigned wins. Shift counts never change the left signedness.
    let source = r#"
fn wide_div(a: U32, b: I64) -> I64 { return a / b; }
fn wide_compare(a: U32, b: I64) -> I32 { return a > b; }
fn reverse_div(a: I64, b: U32) -> I64 { return a / b; }
fn equal_compare(a: I32, b: U32) -> I32 { return a > b; }
fn small_div(a: U8, b: I16) -> I32 { return a / b; }
fn signed_shift(a: I32, b: U32) -> I32 { return a >> b; }
fn widened_signed_shift(a: I32, b: U64) -> I64 { return a >> b; }
fn wide_add(a: U32, b: I64) -> I64 { return a + b; }
"#;
    for opt in [0, 3] {
        let jit = compile(source, opt);
        unsafe {
            let wide_div = entry!(jit, "wide_div", unsafe extern "C" fn(u32, i64) -> i64);
            let wide_compare = entry!(jit, "wide_compare", unsafe extern "C" fn(u32, i64) -> i32);
            let reverse_div = entry!(jit, "reverse_div", unsafe extern "C" fn(i64, u32) -> i64);
            let equal_compare = entry!(jit, "equal_compare", unsafe extern "C" fn(i32, u32) -> i32);
            let small_div = entry!(jit, "small_div", unsafe extern "C" fn(u8, i16) -> i32);
            let signed_shift = entry!(jit, "signed_shift", unsafe extern "C" fn(i32, u32) -> i32);
            let widened_signed_shift = entry!(
                jit,
                "widened_signed_shift",
                unsafe extern "C" fn(i32, u64) -> i64
            );
            let wide_add = entry!(jit, "wide_add", unsafe extern "C" fn(u32, i64) -> i64);
            assert_eq!(wide_div(3_000_000_000, -2), -1_500_000_000);
            assert_eq!(wide_compare(3_000_000_000, -1), 1);
            assert_eq!(reverse_div(-9, 2), -4);
            assert_eq!(equal_compare(-1, 2), 1);
            assert_eq!(small_div(200, -2), -100);
            assert_eq!(signed_shift(-16, 1), -8);
            assert_eq!(widened_signed_shift(-16, 1), -8);
            assert_eq!(wide_add(3_000_000_000, -1), 2_999_999_999);
        }
    }
}

#[test]
fn compounds_and_unsigned_storage_elements_use_the_same_operator_rules() {
    let source = r#"
struct Values { number: U64, words: [U32; 2], }
fn compound(a: U32, b: U32) -> U32 { let mut x: U32 = a; x /= b; return x; }
fn compound_wide(a: U32, b: I64) -> U32 { let mut x: U32 = a; x /= b; return x; }
fn compound_literal(a: U64) -> U64 { let mut x: U64 = a; x /= 3; return x; }
fn array() -> U32 {
    let mut a: [U32; 2] = {};
    a[0] = 3000000000;
    a[1] = 2;
    return a[0] / a[1];
}
fn field() -> U64 {
    let mut a: Values = {};
    let one: U64 = 1;
    a.number = (one << 63) + 4;
    a.number /= 2;
    return a.number;
}
@unsafe
fn reference(a: &mut U64) -> U64 { *a /= 2; return *a; }
@unsafe
fn reference_array(a: &mut [U32; 2]) -> U32 { a[0] /= a[1]; return a[0]; }
"#;
    for opt in [0, 3] {
        let jit = compile(source, opt);
        unsafe {
            let compound = entry!(jit, "compound", unsafe extern "C" fn(u32, u32) -> u32);
            let compound_wide = entry!(jit, "compound_wide", unsafe extern "C" fn(u32, i64) -> u32);
            let compound_literal =
                entry!(jit, "compound_literal", unsafe extern "C" fn(u64) -> u64);
            let array = entry!(jit, "array", unsafe extern "C" fn() -> u32);
            let field = entry!(jit, "field", unsafe extern "C" fn() -> u64);
            let reference = entry!(jit, "reference", unsafe extern "C" fn(*mut u64) -> u64);
            let reference_array = entry!(
                jit,
                "reference_array",
                unsafe extern "C" fn(*mut u32) -> u32
            );
            assert_eq!(compound(3_000_000_000, 2), 1_500_000_000);
            assert_eq!(
                compound_wide(3_000_000_000, -2),
                (-1_500_000_000_i64) as u32
            );
            assert_eq!(
                compound_literal(0x8000_0000_0000_0005),
                0x8000_0000_0000_0005_u64 / 3
            );
            assert_eq!(array(), 1_500_000_000);
            assert_eq!(field(), 0x4000_0000_0000_0002);
            let mut high = 0x8000_0000_0000_0004_u64;
            assert_eq!(reference(&mut high), 0x4000_0000_0000_0002);
            assert_eq!(high, 0x4000_0000_0000_0002);
            let mut values = [3_000_000_000_u32, 2];
            assert_eq!(reference_array(values.as_mut_ptr()), 1_500_000_000);
            assert_eq!(values, [1_500_000_000, 2]);
        }
    }
}

#[test]
fn float_literals_preserve_bits_and_nan_inequality_is_unordered() {
    let source = r#"
fn precise() -> F64 { return 1.0000001; }
fn pi() -> F64 { return 3.141592653589793; }
fn single() -> F32 { return 1.0000001; }
fn not_equal(a: F64, b: F64) -> I32 { return a != b; }
fn equal(a: F64, b: F64) -> I32 { return a == b; }
fn less(a: F64, b: F64) -> I32 { return a < b; }
fn both(a: U32, b: U32) -> I32 {
    if (a > b) && (a != b) { return 1; }
    return 0;
}
"#;
    for opt in [0, 3] {
        let jit = compile(source, opt);
        unsafe {
            let precise = entry!(jit, "precise", unsafe extern "C" fn() -> f64);
            let pi = entry!(jit, "pi", unsafe extern "C" fn() -> f64);
            let single = entry!(jit, "single", unsafe extern "C" fn() -> f32);
            let not_equal = entry!(jit, "not_equal", unsafe extern "C" fn(f64, f64) -> i32);
            let equal = entry!(jit, "equal", unsafe extern "C" fn(f64, f64) -> i32);
            let less = entry!(jit, "less", unsafe extern "C" fn(f64, f64) -> i32);
            let both = entry!(jit, "both", unsafe extern "C" fn(u32, u32) -> i32);
            assert_eq!(precise().to_bits(), 1.0000001_f64.to_bits());
            assert_eq!(pi().to_bits(), std::f64::consts::PI.to_bits());
            assert_eq!(single().to_bits(), 1.0000001_f32.to_bits());
            assert_eq!(not_equal(f64::NAN, f64::NAN), 1);
            assert_eq!(not_equal(f64::NAN, 1.0), 1);
            assert_eq!(not_equal(1.0, f64::NAN), 1);
            assert_eq!(not_equal(1.0, 1.0), 0);
            assert_eq!(not_equal(1.0, 2.0), 1);
            assert_eq!(equal(f64::NAN, f64::NAN), 0);
            assert_eq!(equal(1.0, 1.0), 1);
            assert_eq!(less(f64::NAN, 1.0), 0);
            assert_eq!(less(1.0, 2.0), 1);
            assert_eq!(both(3_000_000_000, 2), 1);
            assert_eq!(both(2, 2), 0);
        }
    }
}
