//! The `--emit-native` backend wrote a RUNNABLE ELF that computed the wrong
//! answer, under "Compiled to native ELF executable!" and exit 0.
//!
//! Of every instance of this repo's design-rule violation, this was the worst:
//! not a refused compile, not a text blob, but a 188-byte executable. Measured
//! before the fix, each one building and running cleanly:
//!
//! ```text
//!     let a = 9; let b = 2; return a / b;   ->  9    (Div emitted NO instruction)
//!     let a = 9; let b = 2; return a % b;   ->  9    (Mod likewise)
//!     let a = 9; let b = 2; return a - b;   ->  0    (both names read `a`)
//!     let a = 9; let b = 2; return b;       ->  9    (ditto)
//! ```
//!
//! Three separate `_ => {}` arms:
//!
//! 1. **`Expr::Ident` ignored its own name** and emitted `mov eax, [rbp-4]` -
//!    the first local - for every identifier. Parameters had no home at all;
//!    nothing spilled `rdi`/`rsi`/... so a function with arguments read stack
//!    garbage.
//! 2. **The `BinaryOp` match** ended in `_ => {}`, so `/`, `%`, all six
//!    comparisons, `&`/`|`/`^` and both shifts emitted nothing and left the
//!    LEFT operand in `eax`.
//! 3. **`emit_stmt` and `emit_expr`** each ended in `_ => {}`, so `if`, `while`,
//!    `for`, `=` and `+=` were dropped silently.
//!
//! **Every case here RUNS the produced binary and compares against a constant.**
//! A test that checked "the ELF is well-formed" passes on all four rows above.
use std::path::PathBuf;
use std::process::Command;

/// Compiles with `--emit-native` and runs the result. `Ok(code)` is the exit
/// status; `Err(diagnostic)` means the backend refused.
fn build_native(name: &str, src: &str) -> Result<i32, String> {
    let dir = std::env::temp_dir().join(format!("y_native_{}_{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let bin = dir.join(name);
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("--emit-native")
        .arg(format!("--output={}", bin.display()))
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
        .output()
        .expect("run Y");
    let all = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if !out.status.success() {
        return Err(all);
    }
    assert!(bin.exists(), "{}: reported success but wrote no file", name);
    Command::new(&bin)
        .status()
        .ok()
        .and_then(|s| s.code())
        .ok_or_else(|| format!("{} did not exit normally", name))
}

/// `return <expr>;` over two locals, so both operands are named - which is what
/// makes the identifier bug visible. Values stay under 128 so the process exit
/// code carries them unchanged.
fn two_local_expr(name: &str, a: i32, b: i32, expr: &str) -> Result<i32, String> {
    build_native(
        name,
        &format!(
            "fn main() -> I32 {{\n    let a: I32 = {};\n    let b: I32 = {};\n    return {};\n}}\n",
            a, b, expr
        ),
    )
}

#[test]
fn every_integer_binary_operator_computes_its_own_operation() {
    // Nine of these emitted no instruction at all and returned `a`. Note `a - b`
    // is in the list for a second reason: `Sub` WAS implemented, and it still
    // returned 0, because both identifiers resolved to `a`. A test that only
    // covered the unimplemented ops would have attributed the bug wrongly.
    let cases: &[(&str, &str, i32)] = &[
        ("op_add", "a + b", 11),
        ("op_sub", "a - b", 7),
        ("op_mul", "a * b", 18),
        ("op_div", "a / b", 4),
        ("op_mod", "a % b", 1),
        ("op_and", "a & b", 0),
        ("op_or", "a | b", 11),
        ("op_xor", "a ^ b", 11),
        ("op_shl", "a << b", 36),
        ("op_shr", "a >> b", 2),
        ("op_gt", "a > b", 1),
        ("op_lt", "a < b", 0),
        ("op_ge", "a >= b", 1),
        ("op_le", "a <= b", 0),
        ("op_eq", "a == b", 0),
        ("op_ne", "a != b", 1),
    ];
    for (name, expr, want) in cases {
        match two_local_expr(name, 9, 2, expr) {
            Ok(got) => assert_eq!(got, *want, "`{}` with a=9, b=2", expr),
            Err(d) => panic!("`{}` was refused but is implemented:\n{}", expr, d),
        }
    }
    // Equality needs a case where it holds, or `==` could be hardcoded to 0.
    assert_eq!(two_local_expr("op_eq_true", 5, 5, "a == b"), Ok(1));
    assert_eq!(two_local_expr("op_le_true", 5, 5, "a <= b"), Ok(1));
}

#[test]
fn identifiers_resolve_to_their_own_local() {
    // The direct probe for bug 1, with three locals so "the first one" and "the
    // last one" are both wrong answers.
    let src = "fn main() -> I32 {\n    \
               let a: I32 = 11;\n    let b: I32 = 22;\n    let c: I32 = 33;\n    \
               return b;\n}\n";
    assert_eq!(
        build_native("ident_mid", src),
        Ok(22),
        "`return b` did not read b"
    );
}

#[test]
fn parameters_are_spilled_and_readable() {
    // Nothing stored the argument registers, so a function with parameters read
    // whatever was on the stack. Asymmetric arguments and a non-commutative
    // operator, so swapping them is also a failure.
    let src = "fn sub2(x: I32, y: I32) -> I32 {\n    return x - y;\n}\n\n\
               fn main() -> I32 {\n    return sub2(30, 8);\n}\n";
    assert_eq!(build_native("params", src), Ok(22), "sub2(30, 8) should be 22");
}

#[test]
fn constructs_this_backend_cannot_encode_are_refused() {
    // The control. Making the emitter emit *something* for everything passes
    // every test above and is exactly the bug that was here. Each case asserts
    // on its OWN diagnostic rather than on "the build failed", so a fixture
    // that is really being rejected by an earlier pass shows up as a failure
    // instead of as a pass.
    let cases: &[(&str, &str, &str)] = &[
        (
            "no_if",
            "fn main() -> I32 {\n    let a: I32 = 1;\n    if a > 0 {\n        return 5;\n    }\n    return 9;\n}\n",
            "`if`",
        ),
        (
            "no_float",
            "fn main() -> F32 {\n    return 1.5;\n}\n",
            "a float literal",
        ),
        (
            "no_unknown_name",
            "fn main() -> I32 {\n    return q;\n}\n",
            "the name `q`",
        ),
        (
            "no_assign",
            "fn main() -> I32 {\n    let a: I32 = 1;\n    a = 2;\n    return a;\n}\n",
            "assignment",
        ),
    ];
    for (name, src, phrase) in cases {
        match build_native(name, src) {
            Ok(code) => panic!(
                "{}: the backend produced a runnable binary (exit {}) for a \
                 construct it cannot encode",
                name, code
            ),
            Err(d) => assert!(
                d.contains("[Native x86-64 Backend]") && d.contains(phrase),
                "{}: refused, but not by the native backend for {} - so this \
                 fixture is stopped by an EARLIER pass and proves nothing:\n{}",
                name,
                phrase,
                d
            ),
        }
    }
}

// ── The datapath is 32 bits, and it used to lie about that ──────────────
//
// `Expr::IntLit` emitted `mov eax, imm32` from an `i64` AST value, and every
// operation runs in `eax`/`ecx`. So a 64-bit type compiled to a runnable ELF
// that computed something else, under the same success banner as everything
// above:
//
//     let a: I64 = 4294967296;                       return a >> 32;  -> 0, want 1
//     let a: I64 = 100000; let b: I64 = 100000; return (a * b) >> 32; -> 0, want 2
//
// The second matters more than the first: it has no large LITERAL in it. The
// values fit in 32 bits and the PRODUCT does not, so a range check on
// literals alone would not have caught it — the declared type is the lie.
//
// This is the `ptx_emitter` integer-width gotcha in its THIRD backend, after
// `llvm_emitter`. Widening the datapath is a feature (REX.W on every
// instruction, `movabs` immediates), not a typo, so the answer is a named
// refusal like every other construct this backend cannot encode.

/// Each case asserts on its own phrase, so a fixture stopped by an earlier
/// check fails instead of passing for the wrong reason.
#[test]
fn sixty_four_bit_types_and_literals_are_refused() {
    let cases = [
        (
            "nat_i64_local",
            "fn main() -> I32 {\n    let a: I64 = 4294967296;\n    return a >> 32;\n}\n",
            "64-bit type",
        ),
        (
            "nat_i64_product",
            "fn main() -> I32 {\n    let a: I64 = 100000;\n    let b: I64 = 100000;\n    return (a * b) >> 32;\n}\n",
            "64-bit type",
        ),
        (
            "nat_i64_param",
            "fn f(x: I64) -> I32 {\n    return x;\n}\nfn main() -> I32 {\n    return 7;\n}\n",
            "64-bit type",
        ),
        (
            "nat_i64_ret",
            "fn f(x: I32) -> I64 {\n    return x;\n}\nfn main() -> I32 {\n    return 7;\n}\n",
            "64-bit type",
        ),
        (
            "nat_wide_literal",
            "fn main() -> I32 {\n    let a: I32 = 3000000000;\n    return a;\n}\n",
            "integer literal",
        ),
    ];
    for (name, src, phrase) in cases {
        match build_native(name, src) {
            Ok(code) => panic!(
                "`{}` produced a RUNNABLE binary (exit {}). The datapath is 32 bits, \
                 so this program's answer is wrong and the banner says otherwise.",
                name, code
            ),
            Err(diag) => assert!(
                diag.contains(phrase),
                "`{}` was refused, but not for being too wide - so this case is \
                 not testing what it claims. Wanted {:?}, got: {}",
                name,
                phrase,
                diag
            ),
        }
    }
}

/// The control, and it carries the weight: refusing every integer type would
/// satisfy every case above and delete the backend. Values that genuinely fit
/// in 32 bits must still compile AND still run to the right answer, including
/// at the boundary.
#[test]
fn thirty_two_bit_programs_still_compile_and_run() {
    let cases = [
        ("nat_ok_small", "fn main() -> I32 {\n    let a: I32 = 9;\n    let b: I32 = 2;\n    return a - b;\n}\n", 7),
        ("nat_ok_i32_max", "fn main() -> I32 {\n    let a: I32 = 2147483647;\n    return a >> 24;\n}\n", 127),
        (
            "nat_ok_params",
            "fn add(x: I32, y: I32) -> I32 {\n    return x + y;\n}\nfn main() -> I32 {\n    return add(20, 3);\n}\n",
            23,
        ),
        ("nat_ok_untyped_let", "fn main() -> I32 {\n    let a = 40;\n    let b = 2;\n    return a + b;\n}\n", 42),
    ];
    for (name, src, want) in cases {
        match build_native(name, src) {
            Ok(code) => assert_eq!(
                code, want,
                "`{}` is entirely within 32 bits and must still run correctly",
                name
            ),
            Err(diag) => panic!("`{}` must still compile, but was refused: {}", name, diag),
        }
    }
}
