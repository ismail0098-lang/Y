//! Rotate recognition must preserve unsigned widths and expression evaluation.
//! Raw IR assertions distinguish the emitter from LLVM's own O3 combines.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::process::Command;
use y::cpu_jit::{host_profile, CpuJit, JitOptions};
use y::{lexer::Lexer, llvm_emitter::LlvmEmitter, parser::Parser};

fn compile(source: &str, opt_level: u8) -> CpuJit {
    CpuJit::compile_with_options(
        source,
        JitOptions {
            opt_level,
            ..Default::default()
        },
    )
    .unwrap_or_else(|error| panic!("O{opt_level}: {error}\n{source}"))
}

fn emitted(source: &str, recognize: bool) -> String {
    let program = Parser::new(Lexer::new(source).tokenize())
        .parse_program()
        .unwrap();
    let mut emitter = LlvmEmitter::new();
    emitter.set_recognize_rotates(recognize);
    let ir = emitter.emit_program(&program, &host_profile());
    assert!(emitter.emit_errors.is_empty(), "{:?}", emitter.emit_errors);
    ir
}

fn body<'a>(ir: &'a str, name: &str) -> &'a str {
    let marker = format!("@{name}(");
    let start = ir
        .lines()
        .find(|line| line.starts_with("define ") && line.contains(&marker))
        .unwrap();
    let position = ir.find(start).unwrap();
    ir[position..].split_once("\n}").unwrap().0
}

macro_rules! entry {
    ($jit:expr, $name:literal, $signature:ty) => {
        std::mem::transmute::<usize, $signature>($jit.function_address($name).unwrap())
    };
}

const ROTATES: &str = r#"
fn left64(x: U64) -> U64 { return (x << 13) | (x >> 51); }
fn right64(x: U64) -> U64 { return (x >> 13) | (x << 51); }
fn commuted64(x: U64) -> U64 { return (x >> 51) | (x << 13); }
fn boundary64(x: U64) -> U64 { return (x << 63) | (x >> 1); }
fn folded64(x: U64) -> U64 { return (x << (8 + 5)) | (x >> (64 - 13)); }
fn left32(x: U32) -> U32 { return (x << 7) | (x >> 25); }
fn right32(x: U32) -> U32 { return (x >> 7) | (x << 25); }
fn boundary32(x: U32) -> U32 { return (x << 1) | (x >> 31); }
fn half32(x: U32) -> U32 { return (x << 16) | (x >> 16); }
fn pointer_width(x: usize) -> usize { return (x >> 19) | (x << 45); }
fn mutable(x: U64) -> U64 {
    let mut value: U64 = x;
    value = value + 3;
    value = (value << 13) | (value >> 51);
    return value;
}
fn byte_promoted(x: U8) -> U32 { return (x << 3) | (x >> 29); }
fn word_promoted(x: U16) -> U32 { return (x >> 5) | (x << 27); }
fn byte_source_width(x: U8) -> U32 { return (x << 3) | (x >> 5); }
fn word_source_width(x: U16) -> U32 { return (x << 5) | (x >> 11); }
"#;

#[test]
fn emitter_uses_funnel_shifts_and_the_option_preserves_generic_lowering() {
    let enabled = emitted(ROTATES, true);
    for (name, intrinsic) in [
        ("left64", "fshl.i64"),
        ("right64", "fshr.i64"),
        ("commuted64", "fshr.i64"),
        ("boundary64", "fshl.i64"),
        ("folded64", "fshl.i64"),
        ("left32", "fshl.i32"),
        ("right32", "fshr.i32"),
        ("boundary32", "fshl.i32"),
        ("half32", "fshl.i32"),
        ("pointer_width", "fshr.i64"),
        ("mutable", "fshl.i64"),
        ("byte_promoted", "fshl.i32"),
        ("word_promoted", "fshr.i32"),
    ] {
        let function = body(&enabled, name);
        assert!(
            function.contains(&format!("@llvm.{intrinsic}(")),
            "{name}: {function}"
        );
        assert!(!function.contains(" = shl ") && !function.contains(" = lshr "));
    }
    let disabled = emitted(ROTATES, false);
    for name in ["left64", "right64", "left32", "right32", "mutable"] {
        let function = body(&disabled, name);
        assert!(!function.contains("@llvm.fsh"), "{name}: {function}");
        assert!(function.contains(" = shl ") && function.contains(" = lshr "));
    }
    for name in ["byte_source_width", "word_source_width"] {
        let function = body(&enabled, name);
        assert!(!function.contains("@llvm.fsh"), "{name}: {function}");
        assert!(function.contains(" = shl i32 ") && function.contains(" = lshr i32 "));
    }
    let jit = CpuJit::compile_with_options(
        ROTATES,
        JitOptions {
            opt_level: 0,
            recognize_rotates: false,
            ..JitOptions::default()
        },
    )
    .unwrap();
    assert!(!body(jit.optimized_ir(), "left64").contains("@llvm.fsh"));
    assert!(body(jit.optimized_ir(), "left64").contains(" = shl i64 "));
    unsafe {
        let left64 = entry!(jit, "left64", unsafe extern "C" fn(u64) -> u64);
        assert_eq!(left64(0x8000_0000_0000_0001), 12288);
    }
}

#[test]
fn native_rotates_match_rust_across_widths_and_high_bits_at_o0_and_o3() {
    for opt in [0, 3] {
        let jit = compile(ROTATES, opt);
        unsafe {
            let left64 = entry!(jit, "left64", unsafe extern "C" fn(u64) -> u64);
            let right64 = entry!(jit, "right64", unsafe extern "C" fn(u64) -> u64);
            let commuted64 = entry!(jit, "commuted64", unsafe extern "C" fn(u64) -> u64);
            let boundary64 = entry!(jit, "boundary64", unsafe extern "C" fn(u64) -> u64);
            let folded64 = entry!(jit, "folded64", unsafe extern "C" fn(u64) -> u64);
            let mutable = entry!(jit, "mutable", unsafe extern "C" fn(u64) -> u64);
            let pointer_width = entry!(jit, "pointer_width", unsafe extern "C" fn(usize) -> usize);
            let left32 = entry!(jit, "left32", unsafe extern "C" fn(u32) -> u32);
            let right32 = entry!(jit, "right32", unsafe extern "C" fn(u32) -> u32);
            let boundary32 = entry!(jit, "boundary32", unsafe extern "C" fn(u32) -> u32);
            let half32 = entry!(jit, "half32", unsafe extern "C" fn(u32) -> u32);
            let mut cases = vec![0, 1, 1 << 63, u64::MAX, 0x8040_2010_0804_0201];
            let mut random = 0x2345_6789_abcd_ef01_u64;
            for _ in 0..128 {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                cases.push(random);
            }
            for x in cases {
                assert_eq!(left64(x), x.rotate_left(13), "O{opt}: {x:x}");
                assert_eq!(right64(x), x.rotate_right(13));
                assert_eq!(commuted64(x), x.rotate_left(13));
                assert_eq!(boundary64(x), x.rotate_left(63));
                assert_eq!(folded64(x), x.rotate_left(13));
                assert_eq!(mutable(x), x.wrapping_add(3).rotate_left(13));
                assert_eq!(pointer_width(x as usize), (x as usize).rotate_right(19));
                let x = x as u32;
                assert_eq!(left32(x), x.rotate_left(7));
                assert_eq!(right32(x), x.rotate_right(7));
                assert_eq!(boundary32(x), x.rotate_left(1));
                assert_eq!(half32(x), x.rotate_left(16));
            }
            let byte_promoted = entry!(jit, "byte_promoted", unsafe extern "C" fn(u8) -> u32);
            let word_promoted = entry!(jit, "word_promoted", unsafe extern "C" fn(u16) -> u32);
            let byte_source_width =
                entry!(jit, "byte_source_width", unsafe extern "C" fn(u8) -> u32);
            let word_source_width =
                entry!(jit, "word_source_width", unsafe extern "C" fn(u16) -> u32);
            for byte in 0..=u8::MAX {
                assert_eq!(byte_promoted(byte), u32::from(byte).rotate_left(3));
                assert_eq!(
                    byte_source_width(byte),
                    (u32::from(byte) << 3) | (u32::from(byte) >> 5)
                );
            }
            for word in [0, 1, 256, 32768, 65535] {
                assert_eq!(word_promoted(word), u32::from(word).rotate_right(5));
                assert_eq!(
                    word_source_width(word),
                    (u32::from(word) << 5) | (u32::from(word) >> 11)
                );
            }
        }
    }
}

const REFUSED: &str = r#"
fn signed(x: I64) -> I64 { return (x << 13) | (x >> 51); }
fn different(x: U64, y: U64) -> U64 { return (x << 13) | (y >> 51); }
fn wrong_sum(x: U64) -> U64 { return (x << 13) | (x >> 50); }
fn same_direction(x: U64) -> U64 { return (x << 13) | (x << 51); }
fn xor(x: U64) -> U64 { return (x << 13) ^ (x >> 51); }
fn dynamic(x: U64, count: U64) -> U64 { return (x << count) | (x >> (64 - count)); }
fn zero(x: U64) -> U64 { return (x << 0) | (x >> 64); }
fn too_large(x: U64) -> U64 { return (x << 65) | (x >> 1); }
fn negative(x: U64) -> U64 { return (x << -1) | (x >> 65); }
@unsafe fn indexed(p: GlobalMemory<U64>) -> U64 { return (p[0] << 13) | (p[0] >> 51); }
@unsafe fn dereferenced(p: &U64) -> U64 { return (*p << 13) | (*p >> 51); }
struct Value { word: U64 }
fn field(p: &Value) -> U64 { return (p.word << 13) | (p.word >> 51); }
@unsafe fn next(p: GlobalMemory<U64>) -> U64 { p[0] = p[0] + 1; return p[0]; }
@unsafe fn calls(p: GlobalMemory<U64>) -> U64 { return (next(p) << 13) | (next(p) >> 51); }
"#;

#[test]
fn signed_near_miss_and_unproved_counts_keep_the_original_ir_contract() {
    let ir = emitted(REFUSED, true);
    for name in [
        "signed",
        "different",
        "wrong_sum",
        "same_direction",
        "xor",
        "dynamic",
        "zero",
        "too_large",
        "negative",
        "indexed",
        "dereferenced",
        "field",
        "calls",
    ] {
        assert!(
            !body(&ir, name).contains("@llvm.fsh"),
            "{name}: {}",
            body(&ir, name)
        );
    }
    assert!(body(&ir, "signed").contains(" = ashr i64 "));
    assert!(body(&ir, "zero").contains(" = lshr i64 "));
    // Do not execute invalid counts: LLVM's existing poison contract is retained.
    for opt in [0, 3] {
        let jit = compile(REFUSED, opt);
        unsafe {
            let signed = entry!(jit, "signed", unsafe extern "C" fn(i64) -> i64);
            let different = entry!(jit, "different", unsafe extern "C" fn(u64, u64) -> u64);
            let wrong_sum = entry!(jit, "wrong_sum", unsafe extern "C" fn(u64) -> u64);
            let dynamic = entry!(jit, "dynamic", unsafe extern "C" fn(u64, u64) -> u64);
            for x in [0, 1, 0x8000_0000_0000_0001, u64::MAX] {
                let sx = x as i64;
                assert_eq!(signed(sx), (sx << 13) | (sx >> 51));
                assert_eq!(different(x, !x), (x << 13) | (!x >> 51));
                assert_eq!(wrong_sum(x), (x << 13) | (x >> 50));
                for count in [1, 13, 32, 63] {
                    assert_eq!(dynamic(x, count), x.rotate_left(count as u32));
                }
            }
        }
    }
}

#[test]
fn repeated_calls_and_index_side_effects_are_evaluated_twice_in_order() {
    let source = format!(
        r#"{REFUSED}
@unsafe fn index(trace: &mut I32) -> I32 {{ let old: I32 = *trace; *trace = old + 1; return old; }}
@unsafe fn indices(p: GlobalMemory<U64>, trace: &mut I32) -> U64 {{
    return (p[index(trace)] << 13) | (p[index(trace)] >> 51);
}}
"#
    );
    assert!(!body(&emitted(&source, true), "indices").contains("@llvm.fsh"));
    for opt in [0, 3] {
        let jit = compile(&source, opt);
        unsafe {
            let calls = entry!(jit, "calls", unsafe extern "C" fn(*mut u64) -> u64);
            let indices = entry!(
                jit,
                "indices",
                unsafe extern "C" fn(*mut u64, *mut i32) -> u64
            );
            for seed in [0_u64, 0x8000_0000_0000_0000, u64::MAX - 1] {
                let mut value = seed;
                let expected = (seed.wrapping_add(1) << 13) | (seed.wrapping_add(2) >> 51);
                assert_eq!(calls(&mut value), expected);
                assert_eq!(value, seed.wrapping_add(2));
            }
            let mut storage = [0x1234_5678_9abc_def0, 0x8000_0000_0000_0000];
            let mut trace = 0;
            assert_eq!(
                indices(storage.as_mut_ptr(), &mut trace),
                (storage[0] << 13) | (storage[1] >> 51)
            );
            assert_eq!(trace, 2);
        }
    }
}

#[test]
fn benchmark_unsigned_mix_uses_rotate_and_preserves_wrapping_results() {
    let source = r#"
@unsafe
fn unsigned_mix(n: I64, seed: U64) -> U64 {
    let mut x: U64 = seed;
    let mut i: I64 = 0;
    while i < n {
        x = x ^ (x >> 12);
        x = x ^ (x << 25);
        x = x ^ (x >> 27);
        x = x * 2685821657736338717;
        x = (x << 13) | (x >> 51);
        i = i + 1;
    }
    return x;
}
"#;
    assert!(body(&emitted(source, true), "unsigned_mix").contains("@llvm.fshl.i64("));
    for opt in [0, 3] {
        let jit = compile(source, opt);
        unsafe {
            let mixed = entry!(jit, "unsigned_mix", unsafe extern "C" fn(i64, u64) -> u64);
            for n in [-1, 0, 1, 2, 17, 1000] {
                for seed in [0, 1, 0x8000_0000_0000_0001, u64::MAX] {
                    let mut expected = seed;
                    for _ in 0..n {
                        expected ^= expected >> 12;
                        expected ^= expected << 25;
                        expected ^= expected >> 27;
                        expected = expected.wrapping_mul(2685821657736338717).rotate_left(13);
                    }
                    assert_eq!(mixed(n, seed), expected, "O{opt}: n={n}, seed={seed:x}");
                }
            }
        }
    }
}

#[test]
fn aot_and_jit_preserve_rotate_widths_and_high_bits() {
    if Command::new("clang").arg("--version").output().is_err() {
        eprintln!("SKIP AOT rotate parity: clang is unavailable");
        return;
    }
    let source = format!(
        r#"{ROTATES}
fn main() -> I32 {{
    let one: U64 = 1;
    let high: U64 = (one << 63) + 1;
    if left64(high) != 12288 {{ return 1; }}
    if right64(high) != 3377699720527872 {{ return 2; }}
    if left32(2147483649) != 192 {{ return 3; }}
    if byte_source_width(128) != 1028 {{ return 4; }}
    if word_promoted(32768) != 1024 {{ return 5; }}
    return 0;
}}
"#
    );
    for opt in [0, 3] {
        assert_eq!(unsafe { compile(&source, opt).run_main().unwrap() }, 0);
        let directory =
            std::env::temp_dir().join(format!("y-rotate-{}-O{opt}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let input = directory.join("rotates.ysu");
        let executable = directory.join("rotates");
        std::fs::write(&input, &source).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&input)
            .arg(format!("-O{opt}"))
            .arg("-o")
            .arg(&executable)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "AOT O{opt}: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let output = Command::new(&executable).output().unwrap();
        assert_eq!(output.status.code(), Some(0), "AOT O{opt}: {output:?}");
        std::fs::remove_dir_all(directory).unwrap();
    }
}
