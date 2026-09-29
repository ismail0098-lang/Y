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
        assert!(!bin.exists(), "{name}: failed compilation still wrote a native binary");
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

#[test]
fn unknown_names_are_rejected_by_frontend_and_native_backend() {
    let src = "fn main() -> I32 {\n    return q;\n}\n";
    // Name resolution now stops this source before native lowering. Exercise
    // the public emitter directly to retain its independent rejection check.
    let ast = y::parser::Parser::new(y::lexer::Lexer::new(src).tokenize())
        .parse_program()
        .expect("parse unresolved identifier");
    let mut emitter = y::native_emitter::NativeEmitter::new();
    emitter.emit_program(&ast);
    assert!(
        emitter.emit_errors.iter().any(|e|
            e.contains("[Native x86-64 Backend]") && e.contains("the name `q`")),
        "native emitter lost its unresolved-name refusal: {:?}",
        emitter.emit_errors
    );
    let diagnostic = build_native("no_unknown_name", src)
        .expect_err("an unresolved identifier must not produce a runnable binary");
    assert!(diagnostic.contains("Undefined variable `q`"), "{diagnostic}");
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

// ── Calls, returns and declared types: the second round ─────────────────
//
// Measured on 9a61d81, each compiling to a runnable ELF under "Compiled to
// native ELF executable!" and exit 0, with the LLVM backend's answer beside it:
//
//     print_int(5); return 0;                    segfault        LLVM: prints 5, exits 0
//     fn write(a) -> a * 2 ... return write(21)  exit 247        LLVM: 42
//     return 7; return 9;                        exit 9          LLVM: 7
//     @safe { return 5; } return 9;              exit 9          LLVM: 5
//     fn main() { let x: I32 = f(); }  (f = 5)   exit 5          LLVM: 0
//     U32: (2^32 - 2) > (2^31 - 1)               0               want 1
//     I8:  (100 + 100) > 0                       1               LLVM: 0 (it wraps to -56)
//     no `fn main` at all                        exit 0          LLVM: link error
//     `fn f` defined twice                       the last one    LLVM: clang redefinition
//
// Each case that runs is compared against a constant; each refusal asserts
// the native backend's own phrase, so a fixture stopped by an earlier pass
// fails instead of passing for the wrong reason.

/// The native backend refused `src` for `phrase`, and `build_native` has
/// already checked that no binary was written.
fn assert_native_refuses(name: &str, src: &str, phrase: &str) {
    match build_native(name, src) {
        Ok(code) => panic!(
            "`{}` produced a RUNNABLE binary (exit {}) where the native backend must \
             refuse it for {:?}",
            name, code, phrase
        ),
        Err(diag) => assert!(
            diag.contains("[Native x86-64 Backend]") && diag.contains(phrase),
            "`{}` was refused, but not by the native backend for {:?} - so this case \
             is not testing what it claims:\n{}",
            name,
            phrase,
            diag
        ),
    }
}

/// A call to a built-in or GPU intrinsic was emitted as `call +0`.
///
/// The type checker refuses a name it does not know, so what reaches this
/// backend undefined is a name it DOES know - and `patch_relocs` skipped any
/// target it could not resolve, leaving `e8 00 00 00 00`: a call to the NEXT
/// INSTRUCTION. The stack unbalances and the final `ret` jumps to garbage.
#[test]
fn a_call_to_a_function_the_program_does_not_define_is_refused() {
    // The smallest reproducer: a runtime function the DEFAULT backend links.
    // It segfaulted here.
    assert_native_refuses(
        "undef_print_int",
        "fn main() -> I32 {\n    print_int(5);\n    return 0;\n}\n",
        "a call to `print_int`",
    );
    // The corpus program that exposed it, read from the repository so the
    // fixture cannot drift from the file.
    let corpus = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/coprocessor_large.ysu"),
    )
    .expect("read tests/coprocessor_large.ysu");
    assert_native_refuses(
        "undef_coprocessor_large",
        &corpus,
        "a call to `rt_nearest_neighbor`",
    );
}

/// The emitter's own check, through the library API. The front end stops an
/// undefined USER name first ("Unknown function"), so the CLI cannot reach
/// this path with one; the emitter must not depend on that.
#[test]
fn the_emitter_itself_refuses_an_undefined_callee() {
    let src = "fn main() -> I32 {\n    return helper(3);\n}\n";
    let ast = y::parser::Parser::new(y::lexer::Lexer::new(src).tokenize())
        .parse_program()
        .expect("parse a call to an undefined function");
    let mut emitter = y::native_emitter::NativeEmitter::new();
    emitter.emit_program(&ast);
    assert!(
        emitter
            .emit_errors
            .iter()
            .any(|e| e.contains("[Native x86-64 Backend]") && e.contains("a call to `helper`")),
        "the native emitter lowered a call to an undefined function: {:?}",
        emitter.emit_errors
    );
}

/// The control that stops "refuse every call": a function defined LATER in
/// the file must still be callable, because the set of definitions is
/// collected before any body is emitted.
#[test]
fn a_forward_call_still_resolves() {
    let src = "fn main() -> I32 {\n    return later(4);\n}\n\n\
               fn later(x: I32) -> I32 {\n    return x * 10;\n}\n";
    assert_eq!(build_native("forward_call", src), Ok(40));
}

/// With no `main`, the entry point's `call main` stayed unresolved, fell
/// into the exit syscall, and the ELF exited 0 having done nothing. The LLVM
/// backend fails the same program at link time.
#[test]
fn a_program_without_main_is_refused() {
    assert_native_refuses(
        "no_main",
        "fn helper() -> I32 {\n    return 1;\n}\n",
        "defines no `main`",
    );
}

/// A syscall stub was registered as `write` and `sys_write` AFTER every
/// function, so it replaced the program's own definition of either name:
/// `write(21)` ran the `write` syscall on file descriptor 21 and exited 247
/// (-EBADF). The answer, and the LLVM backend's, is 42.
#[test]
fn a_function_named_write_is_the_programs_own() {
    for name in ["write", "sys_write"] {
        let src = format!(
            "fn {name}(a: I32) -> I32 {{\n    return a * 2;\n}}\n\n\
             fn main() -> I32 {{\n    return {name}(21);\n}}\n"
        );
        assert_eq!(build_native(&format!("own_{name}"), &src), Ok(42), "`{name}(21)`");
    }
}

/// `return` evaluated its value and carried on, so the LAST return won - the
/// ZK emitter's old `emit_block` bug, in this backend. Every case answers
/// differently under that bug, and each want is the LLVM backend's answer.
/// `ret_in_callee` matters most: the epilogue must hand control back to the
/// CALLER with the stack intact, not merely end the process.
#[test]
fn return_is_a_terminator() {
    let cases: &[(&str, &str, i32)] = &[
        ("ret_twice", "fn main() -> I32 {\n    return 7;\n    return 9;\n}\n", 7),
        (
            "ret_then_let",
            "fn main() -> I32 {\n    return 7;\n    let a: I32 = 9;\n}\n",
            7,
        ),
        (
            "ret_in_safe_block",
            "fn main() -> I32 {\n    @safe {\n        return 5;\n    }\n    return 9;\n}\n",
            5,
        ),
        (
            "ret_after_call",
            "fn id(x: I32) -> I32 {\n    return x;\n}\n\n\
             fn main() -> I32 {\n    let r: I32 = id(4);\n    return r;\n    return 99;\n}\n",
            4,
        ),
        (
            "ret_in_callee",
            "fn f() -> I32 {\n    return 3;\n    return 8;\n}\n\n\
             fn main() -> I32 {\n    let a: I32 = f();\n    return a + 30;\n}\n",
            33,
        ),
    ];
    for (name, src, want) in cases {
        assert_eq!(build_native(name, src), Ok(*want), "`{}`", name);
    }
}

/// Reaching the end of a body returned whatever the last expression left in
/// `eax`. The type checker accepts all four programs, and the LLVM backend
/// answers 0 for the first three (a deliberate `ret i32 0` / `return 0`).
#[test]
fn falling_off_the_end_returns_zero() {
    let cases: &[(&str, &str, i32)] = &[
        (
            "void_main",
            "fn f() -> I32 {\n    return 5;\n}\n\nfn main() {\n    let x: I32 = f();\n}\n",
            0,
        ),
        (
            "void_main_bare_return",
            "fn main() {\n    let x: I32 = 5;\n    return;\n}\n",
            0,
        ),
        (
            "nonvoid_fall_off",
            "fn f() -> I32 {\n    let a: I32 = 5;\n}\n\nfn main() -> I32 {\n    return f();\n}\n",
            0,
        ),
        (
            "bare_return_in_callee",
            "fn g() {\n    let z: I32 = 3;\n    return;\n}\n\n\
             fn main() -> I32 {\n    g();\n    return 6;\n}\n",
            6,
        ),
    ];
    for (name, src, want) in cases {
        assert_eq!(build_native(name, src), Ok(*want), "`{}`", name);
    }
}

/// The datapath is 32-bit SIGNED (`idiv`, `sar`, signed `setcc`) and every
/// value lives in a full register, so an unsigned or sub-word declared type
/// computed a different function, and a float computed integer instructions
/// on its bits. The 64-bit refusal above was one row of this table.
#[test]
fn declared_types_other_than_i32_and_bool_are_refused() {
    let cases: &[(&str, &str, &str)] = &[
        (
            "ty_u32_compare",
            // (2^32 - 2) > (2^31 - 1) is true; the signed compare said 0.
            "fn main() -> I32 {\n    let a: U32 = 2147483647;\n    let b: U32 = a + a;\n    \
             let c: I32 = b > a;\n    return c;\n}\n",
            "unsigned type `U32`",
        ),
        (
            "ty_u32_shift",
            // (2^32 - 2) >> 31 is 1; `sar` gave -1.
            "fn half(x: U32) -> U32 {\n    return x >> 31;\n}\n\n\
             fn main() -> I32 {\n    return 0;\n}\n",
            "unsigned type `U32`",
        ),
        ("ty_u8", "fn main() -> I32 {\n    let a: U8 = 200;\n    return 0;\n}\n", "unsigned type `U8`"),
        ("ty_u16", "fn main() -> I32 {\n    let a: U16 = 60000;\n    return 0;\n}\n", "unsigned type `U16`"),
        (
            "ty_i8_wrap",
            // 100 + 100 wraps to -56 in I8, so `b > 0` is false; this said 1.
            "fn main() -> I32 {\n    let a: I8 = 100;\n    let b: I8 = a + a;\n    \
             let c: I32 = b > 0;\n    return c;\n}\n",
            "sub-word type `I8`",
        ),
        ("ty_i16", "fn main() -> I32 {\n    let a: I16 = 1000;\n    return 0;\n}\n", "sub-word type `I16`"),
        (
            "ty_f32_param",
            "fn sq(x: F32) -> F32 {\n    return x * x;\n}\n\nfn main() -> I32 {\n    return 0;\n}\n",
            "type `F32`",
        ),
        // `Bool` is not a keyword: it reaches here as an unknown type name, and
        // the LLVM backend refuses it as well. The keyword is `bool`.
        ("ty_capital_bool", "fn main() -> Bool {\n    return 3 > 2;\n}\n", "type `Bool`"),
    ];
    for (name, src, phrase) in cases {
        assert_native_refuses(name, src, phrase);
    }
}

/// The control, and it carries the weight: refusing every declared type
/// satisfies the test above and deletes the backend. `bool` - the keyword -
/// must still compile and run, as a return value, a parameter and a `let`.
#[test]
fn bool_still_compiles_and_runs() {
    let cases: &[(&str, &str, i32)] = &[
        ("bool_main", "fn main() -> bool {\n    return 3 > 2;\n}\n", 1),
        (
            "bool_param",
            "fn pick(t: bool) -> bool {\n    return t;\n}\n\n\
             fn main() -> bool {\n    return pick(9 > 5);\n}\n",
            1,
        ),
        (
            "bool_let",
            "fn main() -> bool {\n    let t: bool = 4 < 3;\n    return t;\n}\n",
            0,
        ),
    ];
    for (name, src, want) in cases {
        match build_native(name, src) {
            Ok(code) => assert_eq!(code, *want, "`{}`", name),
            Err(d) => panic!("`{}` uses only `bool` and must still compile:\n{}", name, d),
        }
    }
}

/// The symbol table keeps one address per name, so with two definitions
/// every call reached whichever came LAST. The type checker keeps one
/// signature per name and does not say so either.
#[test]
fn a_second_definition_of_a_function_is_refused() {
    assert_native_refuses(
        "dup_fn",
        "fn f() -> I32 {\n    return 1;\n}\n\nfn f() -> I32 {\n    return 2;\n}\n\n\
         fn main() -> I32 {\n    return f();\n}\n",
        "`fn f` is defined twice",
    );
}

/// Under `@unsafe` the front end allows a `let` with no initializer, and this
/// backend stored whatever `eax` held: `let q: I32 = 77; let a: I32; return
/// a;` exited 77 - another variable's value. With no assignment here, such a
/// local can never be given a value of its own.
#[test]
fn an_uninitialized_let_is_refused() {
    assert_native_refuses(
        "uninit_let",
        "@unsafe\nfn main() -> I32 {\n    let q: I32 = 77;\n    let a: I32;\n    return a;\n}\n",
        "a `let` without an initializer",
    );
}
