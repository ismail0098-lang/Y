//! Ordinary Q formats on the LLVM backend, executed against an independent
//! oracle.
//!
//! A `Q16.16` outside `@ZeroDrift` was given `emit_type`'s `i32` default and
//! computed as an integer: `let x: Q16.16 = 1.5; if x > 1.0 { return 7; }
//! return 3;` built, ran and exited **3**. A Q value is now the integer
//! `value * 2^frac` in storage of the format's width, and `fixed.rs` keeps the
//! arithmetic there:
//!
//! * literals, `*` and `/` round to nearest, ties away from zero (the rule
//!   `@ZeroDrift` already used for its terms);
//! * a result outside the format's range, and a division by zero, trap;
//! * what has no fixed-point lowering is refused by name.
//!
//! The oracle below is this file's own arithmetic on `i128`/`u128`, written
//! independently of the emitter (it rounds with `2r >= b`, the emitter with
//! `r >= b - r`), and every case runs at `-O0` and `-O2`.
use std::fmt::Write as _;
use std::os::unix::process::ExitStatusExt;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

#[derive(Clone, Copy)]
struct Fmt {
    int: u32,
    frac: u32,
}

impl Fmt {
    fn bits(self) -> u32 {
        self.int + self.frac
    }
    fn name(self) -> String {
        format!("Q{}.{}", self.int, self.frac)
    }
    fn lo(self) -> i128 {
        -(1i128 << (self.bits() - 1))
    }
    fn hi(self) -> i128 {
        (1i128 << (self.bits() - 1)) - 1
    }
    /// The C type the harness passes and receives. Formats narrower than 32
    /// bits travel as `int32_t` and are truncated on receipt, so the test does
    /// not depend on who sign-extends a small integer across the C ABI.
    fn c_type(self) -> &'static str {
        if self.bits() == 64 {
            "int64_t"
        } else {
            "int32_t"
        }
    }
    fn c_narrow(self) -> &'static str {
        match self.bits() {
            8 => "(int8_t)",
            16 => "(int16_t)",
            _ => "",
        }
    }
}

const FORMATS: [Fmt; 5] = [
    Fmt { int: 4, frac: 4 },
    Fmt { int: 8, frac: 8 },
    Fmt { int: 16, frac: 16 },
    Fmt { int: 32, frac: 32 },
    Fmt { int: 16, frac: 48 },
];

/// `n / d` to nearest, ties away from zero.
fn round_div(n: i128, d: i128) -> i128 {
    let (a, b) = (n.unsigned_abs(), d.unsigned_abs());
    let (q, r) = (a / b, a % b);
    let magnitude = (if 2 * r >= b { q + 1 } else { q }) as i128;
    if (n < 0) != (d < 0) {
        -magnitude
    } else {
        magnitude
    }
}

/// The oracle: `None` where the program must trap.
fn oracle(op: &str, a: i128, b: i128, f: Fmt) -> Option<i128> {
    let exact = match op {
        "add" => a + b,
        "sub" => a - b,
        "mul" => round_div(a * b, 1i128 << f.frac),
        "div" if b == 0 => return None,
        "div" => round_div(a << f.frac, b),
        "neg" => -a,
        _ => unreachable!(),
    };
    (f.lo()..=f.hi()).contains(&exact).then_some(exact)
}

/// A deterministic spread of raw values: every magnitude from one ulp to
/// the whole range, both signs, and the edges.
fn raw_values(f: Fmt, count: usize) -> Vec<i128> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15 ^ (f.bits() as u64);
    let mut out = vec![0, 1, -1, 2, -2, f.lo(), f.hi(), f.lo() + 1, f.hi() - 1];
    out.extend([1i128 << f.frac, -(1i128 << f.frac), 1i128 << (f.frac - 1), -(1i128 << (f.frac - 1))]);
    while out.len() < count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let shift = (state >> 58) as u32 % f.bits();
        let raw = ((state as i64) >> (64 - f.bits())) as i128 >> shift;
        out.push(raw.clamp(f.lo(), f.hi()));
    }
    out
}

fn clang_available() -> bool {
    Command::new("clang").arg("--version").output().is_ok()
}

fn emit(source: &str) -> (String, Vec<String>) {
    let program = y::parser::Parser::new(y::lexer::Lexer::new(source).tokenize())
        .parse_program()
        .expect("parse");
    let mut checker = y::type_checker::TypeChecker::new();
    checker.check_program(&program);
    assert!(checker.errors.is_empty(), "{:?}\n{}", checker.errors, source);
    let mut emitter = y::llvm_emitter::LlvmEmitter::new();
    let ir = emitter.emit_program(&program, &y::sentinel::HardwareProfile::default());
    (ir, emitter.emit_errors)
}

/// Link the module with a C `main` at -O0 and -O2 and run it; returns the
/// exit status of each run.
fn run_with_c(tag: &str, source: &str, c_main: &str) -> Option<Vec<std::process::ExitStatus>> {
    let (ir, errors) = emit(source);
    assert!(errors.is_empty(), "{:?}", errors);
    if !clang_available() {
        eprintln!("SKIP {tag}: no clang on this machine, so this test checked NOTHING");
        return None;
    }
    let dir = pinned::pinned_scratch(&format!("fixed_{tag}"), pinned::SM_PINNED);
    let ll = dir.join("q.ll");
    let c = dir.join("main.c");
    let bin = dir.join("q");
    std::fs::write(&ll, &ir).unwrap();
    std::fs::write(&c, format!("#include <stdint.h>\n#include <stdio.h>\n{c_main}")).unwrap();
    let mut statuses = Vec::new();
    for level in ["-O0", "-O2"] {
        let built = Command::new("clang")
            .args([level, "-Wno-override-module"])
            .arg(&ll)
            .arg(&c)
            .arg("-o")
            .arg(&bin)
            .output()
            .unwrap();
        assert!(built.status.success(), "{}\n{ir}", String::from_utf8_lossy(&built.stderr));
        let ran = Command::new(&bin).output().unwrap();
        if !ran.status.success() && ran.status.code().is_some() {
            panic!("{tag} at {level}: {}", String::from_utf8_lossy(&ran.stdout));
        }
        statuses.push(ran.status);
    }
    Some(statuses)
}

/// A C literal for any 64-bit value; the most negative one has no literal.
fn c_lit(v: i128) -> String {
    if v == i64::MIN as i128 {
        "(-9223372036854775807LL - 1)".into()
    } else {
        format!("{v}LL")
    }
}

#[test]
fn arithmetic_matches_the_oracle_in_every_format() {
    for f in FORMATS {
        let q = f.name();
        let source = format!(
            "fn add(a: {q}, b: {q}) -> {q} {{ return a + b; }}\n\
             fn sub(a: {q}, b: {q}) -> {q} {{ return a - b; }}\n\
             fn mul(a: {q}, b: {q}) -> {q} {{ return a * b; }}\n\
             fn div(a: {q}, b: {q}) -> {q} {{ return a / b; }}\n\
             fn neg(a: {q}, b: {q}) -> {q} {{ return -a; }}\n\
             fn lt(a: {q}, b: {q}) -> I32 {{ if a < b {{ return 1; }} return 0; }}\n\
             fn eq(a: {q}, b: {q}) -> I32 {{ if a == b {{ return 1; }} return 0; }}\n\
             fn gt_lit(a: {q}, b: {q}) -> I32 {{ if a > 1.5 {{ return 1; }} return 0; }}\n\
             fn le_neg_lit(a: {q}, b: {q}) -> I32 {{ if a <= -0.75 {{ return 1; }} return 0; }}\n"
        );
        let values = raw_values(f, 64);
        let pairs = values.len() * values.len();
        let (t, n) = (f.c_type(), f.c_narrow());
        let mut c = String::from("struct qcase { int64_t a, b, want; };\n");
        let mut body = String::from("int main(void) {\n  int bad = 0;\n");
        for op in ["add", "sub", "mul", "div", "neg", "lt", "eq", "gt_lit", "le_neg_lit"] {
            let comparison = matches!(op, "lt" | "eq" | "gt_lit" | "le_neg_lit");
            writeln!(c, "extern {} {op}({t}, {t});", if comparison { "int32_t" } else { t }).unwrap();
            let mut rows = Vec::new();
            for &a in &values {
                for &b in &values {
                    let want = match op {
                        "lt" => Some((a < b) as i128),
                        "eq" => Some((a == b) as i128),
                        // 1.5 and -0.75 are exact in every format here.
                        "gt_lit" => Some((a > 3 << (f.frac - 1)) as i128),
                        "le_neg_lit" => Some((a <= -(3 << (f.frac - 2))) as i128),
                        _ => oracle(op, a, b, f),
                    };
                    if let Some(w) = want {
                        rows.push(format!("{{{}, {}, {}}}", c_lit(a), c_lit(b), c_lit(w)));
                    }
                }
            }
            // Enough of every operation lands in range for the check to mean
            // something; the out-of-range cases are the trap test's business.
            assert!(rows.len() * 8 >= pairs, "{q} {op}: only {} in-range cases", rows.len());
            writeln!(c, "static const struct qcase {op}_cases[] = {{\n{}\n}};", rows.join(",\n")).unwrap();
            let narrow = if comparison { "" } else { n };
            writeln!(
                body,
                "  for (unsigned i = 0; i < sizeof {op}_cases / sizeof {op}_cases[0]; i++) {{\n\
                 \x20   const struct qcase *k = &{op}_cases[i];\n\
                 \x20   if ((int64_t){narrow}{op}(({t})k->a, ({t})k->b) != k->want) {{\n\
                 \x20     printf(\"{op} %lld %lld\\n\", (long long)k->a, (long long)k->b); bad = 1;\n    }}\n  }}"
            )
            .unwrap();
        }
        body.push_str("  return bad;\n}\n");
        c.push_str(&body);
        if let Some(statuses) = run_with_c(&format!("ops_{}_{}", f.int, f.frac), &source, &c) {
            for s in statuses {
                assert!(s.success(), "{q}: {s:?}");
            }
        }
    }
}

#[test]
fn literals_round_to_nearest_with_ties_away_from_zero() {
    // (format, literal, raw value it must quantise to)
    let cases: [(Fmt, &str, i128); 9] = [
        (FORMATS[2], "1.5", 98304),
        (FORMATS[2], "0.1", 6554),       // 6553.6
        (FORMATS[2], "-0.1", -6554),
        (FORMATS[2], "3", 196608),
        (FORMATS[1], "0.001953125", 1),  // exactly half an ulp of Q8.8
        (FORMATS[1], "-0.001953125", -1),
        (FORMATS[1], "0.0019", 0),       // just under half an ulp
        (FORMATS[0], "-8", -128),        // the most negative Q4.4
        (FORMATS[3], "0.75", 3221225472),
    ];
    let mut source = String::new();
    let mut c = String::from("int main(void) {\n  int bad = 0;\n");
    for (i, (f, lit, raw)) in cases.iter().enumerate() {
        let q = f.name();
        // Through a `let`, and returned directly: each is its own lowering.
        writeln!(source, "fn c{i}() -> {q} {{ let v: {q} = {lit}; return v; }}").unwrap();
        writeln!(source, "fn d{i}() -> {q} {{ return {lit}; }}").unwrap();
        for g in ["c", "d"] {
            writeln!(c, "  extern {} {g}{i}(void);", f.c_type()).unwrap();
            writeln!(
                c,
                "  if ({}{g}{i}() != ({}){raw}LL) {{ printf(\"{g}{i}\\n\"); bad = 1; }}",
                f.c_narrow(),
                f.c_type()
            )
            .unwrap();
        }
    }
    c.push_str("  return bad;\n}\n");
    if let Some(statuses) = run_with_c("literals", &source, &c) {
        assert!(statuses.iter().all(|s| s.success()), "{statuses:?}");
    }
}

#[test]
fn statements_keep_the_scale_through_lets_assignments_and_calls() {
    let source = "fn twice(x: Q16.16) -> Q16.16 { return x + x; }\n\
        fn walk() -> Q16.16 {\n    let mut y: Q16.16 = 1.5;\n    y += 0.25;\n    y *= 2.0;\n    \
        y = y - 0.5;\n    let z = y * y;\n    let w: Q16.16 = twice(z) / 3;\n    \
        let v: Q16.16 = twice(1.25) + twice(-0.5);\n    return v - w;\n}\n";
    // y = 1.75 -> 3.5 -> 3.0; z = 9.0; w = 18 / 3 = 6.0; v = 2.5 - 1.0 = 1.5;
    // result -4.5. The literal arguments are what an unscaled `fptosi` breaks:
    // a Q variable's raw value passes through unchanged either way.
    let c = "extern int32_t walk(void);\nint main(void) { return walk() == -294912 ? 0 : 1; }\n";
    if let Some(statuses) = run_with_c("statements", source, c) {
        assert!(statuses.iter().all(|s| s.success()), "{statuses:?}");
    }
}

/// Manual §10 Example 4, the fixed-point filter: Q fields in a struct
/// literal, read and written through `&mut`, combined with a literal.
#[test]
fn struct_fields_keep_the_scale() {
    let source = "struct FilterState {\n    coefficient: Q32.32,\n    prev_value: Q32.32,\n}\n\n\
        fn apply_filter(state: &mut FilterState, signal: Q32.32) -> Q32.32 {\n    \
        let filtered: Q32.32 = (signal * state.coefficient) + (state.prev_value * (1.0 - state.coefficient));\n    \
        state.prev_value = filtered;\n    return filtered;\n}\n\n\
        fn run() -> Q32.32 {\n    let mut filter = FilterState {\n        coefficient: 0.25,\n        prev_value: 0.0,\n    };\n    \
        let first: Q32.32 = apply_filter(&mut filter, 42.0);\n    \
        let second: Q32.32 = apply_filter(&mut filter, 2.0);\n    \
        return first + second + filter.prev_value;\n}\n";
    // first = 42 * 0.25 = 10.5; second = 2 * 0.25 + 10.5 * 0.75 = 8.375;
    // prev_value = 8.375; total 27.25, exact in Q32.32.
    let c = "extern int64_t run(void);\nint main(void) { return run() == (int64_t)(27.25 * 4294967296.0) ? 0 : 1; }\n";
    if let Some(statuses) = run_with_c("struct", source, c) {
        assert!(statuses.iter().all(|s| s.success()), "{statuses:?}");
    }
}

#[test]
fn overflow_and_division_by_zero_trap_instead_of_wrapping() {
    let source = "fn add(a: Q16.16, b: Q16.16) -> Q16.16 { return a + b; }\n\
        fn mul(a: Q16.16, b: Q16.16) -> Q16.16 { return a * b; }\n\
        fn div(a: Q16.16, b: Q16.16) -> Q16.16 { return a / b; }\n\
        fn neg(a: Q16.16) -> Q16.16 { return -a; }\n\
        fn big(a: Q32.32, b: Q32.32) -> Q32.32 { return a * b; }\n";
    let calls = [
        "add(2147483647, 1)",
        "mul(256 * 65536, 256 * 65536)",
        "div(65536, 0)",
        "neg(-2147483647 - 1)",
        "big(INT64_C(1) << 62, INT64_C(1) << 40)",
    ];
    for (i, call) in calls.iter().enumerate() {
        let c = format!(
            "extern int32_t add(int32_t, int32_t);\nextern int32_t mul(int32_t, int32_t);\n\
             extern int32_t div(int32_t, int32_t);\nextern int32_t neg(int32_t);\n\
             extern int64_t big(int64_t, int64_t);\n\
             int main(void) {{ volatile int64_t r = {call}; printf(\"%lld\\n\", (long long)r); return 0; }}\n"
        );
        if let Some(statuses) = run_with_c(&format!("trap{i}"), source, &c) {
            for s in statuses {
                let signal = s.signal();
                assert!(
                    matches!(signal, Some(4) | Some(5)),
                    "`{call}` must trap (SIGILL/SIGTRAP), got {s:?}"
                );
            }
        }
    }
    // The control: the same functions in range do not trap.
    let c = "extern int32_t add(int32_t, int32_t);\nextern int32_t div(int32_t, int32_t);\n\
             int main(void) { return add(65536, 65536) == 131072 && div(65536, 131072) == 32768 ? 0 : 1; }\n";
    if let Some(statuses) = run_with_c("trap_control", source, c) {
        assert!(statuses.iter().all(|s| s.success()), "{statuses:?}");
    }
}

#[test]
fn uses_without_a_fixed_point_lowering_are_refused_by_name() {
    for (body, message) in [
        ("struct P { v: [Q16.16; 2] }\nfn f() -> I32 { return 0; }", "`P.v` is an array of Q16.16"),
        ("fn f() -> I32 { let x: Q16.16 = 1.5; match x { y => print_int(1) } return 0; }", "`match` on a Q16.16"),
        ("fn f() -> I32 { let x: Q16.16 = 1.5; print_int(x); return 0; }", "`print_int` would receive a Q16.16"),
        ("fn f() -> I32 { let x: Q8.8 = 200.0; return 0; }", "outside Q8.8's representable range"),
        ("fn f() -> I32 { let x: Q10.10 = 1.0; return 0; }", "Q10.10 needs 20-bit storage"),
        (
            "fn f() -> I32 {\n    @ZeroDrift\n    let acc: Q32.32 = 0.0;\n    acc += 1.0;\n    let y: Q32.32 = acc;\n    return 0;\n}",
            "@ZeroDrift accumulator `acc`",
        ),
    ] {
        let program = y::parser::Parser::new(y::lexer::Lexer::new(body).tokenize())
            .parse_program()
            .expect("parse");
        let mut emitter = y::llvm_emitter::LlvmEmitter::new();
        emitter.emit_program(&program, &y::sentinel::HardwareProfile::default());
        assert!(
            emitter.emit_errors.iter().any(|e| e.contains(message)),
            "{body}\nwanted `{message}` in {:?}",
            emitter.emit_errors
        );
    }
}

/// The program that ran to the wrong answer, through the CLI.
#[test]
fn the_reported_comparison_now_holds() {
    let dir = pinned::pinned_scratch("fixed_cli", pinned::SM_PINNED);
    let src = dir.join("q.ysu");
    std::fs::write(
        &src,
        "fn main() -> I32 {\n    let x: Q16.16 = 1.5;\n    if x > 1.0 {\n        return 7;\n    }\n    return 3;\n}\n",
    )
    .unwrap();
    let bin = dir.join("q");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .current_dir(&dir)
        .output()
        .expect("run Y");
    if !bin.exists() {
        if !clang_available() {
            eprintln!("SKIP: no clang on this machine, so this test checked NOTHING");
            return;
        }
        panic!("did not build:\n{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    }
    assert_eq!(Command::new(&bin).status().unwrap().code(), Some(7));
}

/// A 64-bit format divides in `i128`, which lowers to a runtime helper
/// (`__udivti3`); the in-process JIT has to resolve it as well as the linker.
#[test]
fn sixty_four_bit_division_runs_in_the_jit() {
    let dir = pinned::pinned_scratch("fixed_jit", pinned::SM_PINNED);
    let src = dir.join("jd.ysu");
    std::fs::write(
        &src,
        "fn main() -> I32 {\n    let a: Q32.32 = 7.5;\n    let b: Q32.32 = 2.5;\n    let c: Q32.32 = a / b;\n    \
         if c == 3.0 {\n        return 9;\n    }\n    return 1;\n}\n",
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg("--target=jit")
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if text.contains("not compiled into this binary") {
        eprintln!("SKIP: the CPU JIT is not in this build, so this test checked NOTHING");
        return;
    }
    assert_eq!(out.status.code(), Some(9), "{text}");
}
