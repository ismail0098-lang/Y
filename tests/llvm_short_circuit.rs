//! Logical operators must preserve skipped side effects and evaluation order.
//! Bounds checks in RHS array accesses also create basic blocks, so nested
//! logical values cannot guess the PHI predecessor from their initial label.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
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

const EFFECTS: &str = r#"
@unsafe
fn record(trace: &mut I32, digit: I32, result: bool) -> bool {
    *trace = *trace * 10 + digit;
    return result;
}
@unsafe
fn conjunction(trace: &mut I32, a: bool, b: bool) -> bool {
    return record(trace, 1, a) && record(trace, 2, b);
}
@unsafe
fn disjunction(trace: &mut I32, a: bool, b: bool) -> bool {
    return record(trace, 1, a) || record(trace, 2, b);
}
@unsafe
fn nested(trace: &mut I32, a: bool, b: bool, c: bool, d: bool) -> bool {
    return (record(trace, 1, a) || record(trace, 2, b)) &&
           (record(trace, 3, c) || record(trace, 4, d));
}
@unsafe
fn nested_rhs(trace: &mut I32, a: bool, b: bool, c: bool) -> bool {
    return record(trace, 1, a) ||
           (record(trace, 2, b) && record(trace, 3, c));
}
"#;

#[test]
fn pointer_side_effects_run_once_in_left_to_right_short_circuit_order() {
    for opt in [0, 3] {
        let jit = compile(EFFECTS, opt);
        unsafe {
            let conjunction = entry!(
                jit,
                "conjunction",
                unsafe extern "C" fn(*mut i32, bool, bool) -> bool
            );
            let disjunction = entry!(
                jit,
                "disjunction",
                unsafe extern "C" fn(*mut i32, bool, bool) -> bool
            );
            let nested = entry!(
                jit,
                "nested",
                unsafe extern "C" fn(*mut i32, bool, bool, bool, bool) -> bool
            );
            let nested_rhs = entry!(
                jit,
                "nested_rhs",
                unsafe extern "C" fn(*mut i32, bool, bool, bool) -> bool
            );
            for (a, b, expected_result, expected_trace) in [
                (false, false, false, 1),
                (false, true, false, 1),
                (true, false, false, 12),
                (true, true, true, 12),
            ] {
                let mut trace = 0;
                assert_eq!(conjunction(&mut trace, a, b), expected_result);
                assert_eq!(trace, expected_trace, "O{opt}: conjunction({a}, {b})");
            }
            for (a, b, expected_result, expected_trace) in [
                (false, false, false, 12),
                (false, true, true, 12),
                (true, false, true, 1),
                (true, true, true, 1),
            ] {
                let mut trace = 0;
                assert_eq!(disjunction(&mut trace, a, b), expected_result);
                assert_eq!(trace, expected_trace, "O{opt}: disjunction({a}, {b})");
            }
            for (a, b, c, d, expected_result, expected_trace) in [
                (false, false, true, true, false, 12),
                (true, false, true, false, true, 13),
                (true, false, false, true, true, 134),
                (false, true, false, true, true, 1234),
                (false, true, false, false, false, 1234),
            ] {
                let mut trace = 0;
                assert_eq!(nested(&mut trace, a, b, c, d), expected_result);
                assert_eq!(trace, expected_trace, "O{opt}: nested");
            }
            for (a, b, c, expected_result, expected_trace) in [
                (true, true, true, true, 1),
                (false, false, true, false, 12),
                (false, true, false, false, 123),
                (false, true, true, true, 123),
            ] {
                let mut trace = 0;
                assert_eq!(nested_rhs(&mut trace, a, b, c), expected_result);
                assert_eq!(trace, expected_trace, "O{opt}: nested RHS");
            }
        }
    }
}

#[test]
fn nested_boolean_results_have_the_right_phi_inputs_for_every_truth_case() {
    let source = r#"
fn first(a: bool, b: bool, c: bool, d: bool) -> bool { return (a && b) || (c && d); }
fn second(a: bool, b: bool, c: bool, d: bool) -> bool { return a && (b || (c && d)); }
fn third(a: bool, b: bool, c: bool, d: bool) -> bool { return !(a || b) && (c || !d); }
fn widened(a: bool, b: bool) -> I64 { return (a && b) == true; }
"#;
    for opt in [0, 3] {
        let jit = compile(source, opt);
        unsafe {
            let first = entry!(
                jit,
                "first",
                unsafe extern "C" fn(bool, bool, bool, bool) -> bool
            );
            let second = entry!(
                jit,
                "second",
                unsafe extern "C" fn(bool, bool, bool, bool) -> bool
            );
            let third = entry!(
                jit,
                "third",
                unsafe extern "C" fn(bool, bool, bool, bool) -> bool
            );
            let widened = entry!(jit, "widened", unsafe extern "C" fn(bool, bool) -> i64);
            for mask in 0..16 {
                let a = mask & 1 != 0;
                let b = mask & 2 != 0;
                let c = mask & 4 != 0;
                let d = mask & 8 != 0;
                assert_eq!(
                    first(a, b, c, d),
                    (a && b) || (c && d),
                    "O{opt}, mask {mask}"
                );
                assert_eq!(
                    second(a, b, c, d),
                    a && (b || (c && d)),
                    "O{opt}, mask {mask}"
                );
                assert_eq!(
                    third(a, b, c, d),
                    !(a || b) && (c || !d),
                    "O{opt}, mask {mask}"
                );
                assert_eq!(widened(a, b), i64::from(a && b));
            }
        }
    }
}

const INDEX_EFFECTS: &str = r#"
@unsafe
fn index(counter: &mut I32, result: I32) -> I32 {
    *counter = *counter + 1;
    return result;
}
@unsafe
fn indexed_and(counter: &mut I32, flags: &mut [bool; 2], left: bool, position: I32) -> bool {
    return left && flags[index(counter, position)];
}
@unsafe
fn indexed_or(counter: &mut I32, flags: &mut [bool; 2], left: bool, position: I32) -> bool {
    return left || flags[index(counter, position)];
}
@unsafe
fn local(left: bool) -> I32 {
    let mut counter: I32 = 0;
    let mut flags: [bool; 2] = {};
    flags[0] = true;
    let result: bool = left && flags[index(&mut counter, 0)];
    let widened: I32 = result == true;
    return counter * 10 + widened;
}
"#;

#[test]
fn rhs_array_address_evaluation_and_bounds_checks_are_skipped_together() {
    for opt in [0, 3] {
        let jit = compile(INDEX_EFFECTS, opt);
        unsafe {
            let indexed_and = entry!(
                jit,
                "indexed_and",
                unsafe extern "C" fn(*mut i32, *mut bool, bool, i32) -> bool
            );
            let indexed_or = entry!(
                jit,
                "indexed_or",
                unsafe extern "C" fn(*mut i32, *mut bool, bool, i32) -> bool
            );
            let local = entry!(jit, "local", unsafe extern "C" fn(bool) -> i32);
            let mut counter = 0;
            let mut flags = [false, true];
            assert!(!indexed_and(&mut counter, flags.as_mut_ptr(), false, 99));
            assert!(indexed_or(&mut counter, flags.as_mut_ptr(), true, -1));
            assert_eq!(counter, 0);
            assert!(indexed_and(&mut counter, flags.as_mut_ptr(), true, 1));
            assert!(!indexed_and(&mut counter, flags.as_mut_ptr(), true, 0));
            assert!(!indexed_or(&mut counter, flags.as_mut_ptr(), false, 0));
            assert!(indexed_or(&mut counter, flags.as_mut_ptr(), false, 1));
            assert_eq!(counter, 4);
            assert_eq!(flags, [false, true]);
            assert_eq!(local(false), 0);
            assert_eq!(local(true), 11);
        }
    }
}

fn aot_exit(source: &str, opt_level: u8) -> i32 {
    static SALT: AtomicUsize = AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "y-logical-{}-{}-O{opt_level}",
        std::process::id(),
        SALT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let input = directory.join("logical.ysu");
    let executable = directory.join("logical");
    std::fs::write(&input, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&input)
        .arg(format!("-O{opt_level}"))
        .arg("-o")
        .arg(&executable)
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "AOT O{opt_level}: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(&executable).output().unwrap();
    let code = output
        .status
        .code()
        .expect("AOT logical program terminated by a signal");
    std::fs::remove_dir_all(directory).unwrap();
    code
}

#[test]
fn aot_and_jit_share_short_circuit_side_effect_and_local_storage_semantics() {
    if Command::new("clang").arg("--version").output().is_err() {
        eprintln!("SKIP AOT parity: clang is unavailable");
        return;
    }
    let source = format!(
        "{EFFECTS}\n{INDEX_EFFECTS}\n{}",
        r#"
@unsafe
fn main() -> I32 {
    let mut trace: I32 = 0;
    let first: bool = conjunction(&mut trace, false, true);
    if first || trace != 1 { return 1; }
    let second: bool = disjunction(&mut trace, true, false);
    if !second || trace != 11 { return 2; }
    if local(false) != 0 || local(true) != 11 { return 3; }
    let mut flags: [bool; 2] = {};
    let mut counter: I32 = 0;
    if indexed_and(&mut counter, &mut flags, false, 99) { return 4; }
    if !indexed_or(&mut counter, &mut flags, true, -1) { return 5; }
    if counter != 0 { return 6; }
    return 0;
}
"#
    );
    for opt in [0, 3] {
        let jit = compile(&source, opt);
        assert_eq!(unsafe { jit.run_main().unwrap() }, 0);
        assert_eq!(aot_exit(&source, opt), 0);
    }
}
