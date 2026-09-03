//! `@require` is a hardware gate that gated nothing.
//!
//! Measured at 5ff9f40, before `src/require.rs` existed:
//!
//! * §9.1 of the language reference documented it as terminating compilation
//!   with `error[R0001]: hardware requirement unsatisfied`. **That string
//!   appeared nowhere in the compiler.**
//! * §2 named `sentinel.rs` as the module that "matches hardware constraints
//!   specified by `@require` decorators against physical microarchitectural
//!   capabilities". **`sentinel.rs` contained the string `require` zero
//!   times.**
//! * `@require(1 == 0)` compiled and emitted PTX, **exit 0**.
//! * Its **sole** reader scanned the condition for an identifier containing
//!   `avx512` and set a local `target_is_cpu` - **written and never read**.
//!   rustc's dead-code lint cannot see that, because the write goes through an
//!   `&mut` parameter, so the warning census could not have found it either.
//! * There are **three syntactic sites and only one retained it**: on a
//!   `kernel` it is stored; on a `fn` (the form §9.1's own example used) and on
//!   an `impl` (the form `self_hosted/lib.ysu` used three times) it parsed and
//!   was **dropped on the floor**, because `KernelDecl` is the only node with a
//!   field for it.
//!
//! A directive whose entire purpose is to refuse, refusing nothing, is the
//! `@zk_target(scheme = "plonkish")` shape. Unlike `@hdl_emit` there is no
//! missing backend behind it - `sentinel` already probes exactly these
//! capabilities - so the fix is to evaluate the condition, not to refuse the
//! directive.
//!
//! **The load-bearing test here is `the_condition_is_actually_evaluated`.** A
//! gate that always refuses satisfies every assertion about a refusal and is
//! useless; a gate that always accepts is the bug being fixed. Only a
//! biconditional on the SAME feature separates them, and it is written to be
//! machine-independent: the refusal message reports the value this host has, so
//! the test reads that value back and asserts the requirement passes at it and
//! fails one above it. That is a boundary one unit wide, on whatever hardware
//! the suite happens to run.

use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Compiles `source` in a per-test scratch directory.
///
/// The `tag` is in the SIGNATURE, not left to the caller: a pid-only temp
/// directory shared between tests in one binary is a race this repository has
/// hit six times. `--emit-ptx` writes its artifact next to the source, which is
/// the other reason this must not run in `tests/`.
fn compile(tag: &str, source: &str, flag: &str) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("y_require_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("p.ysu");
    std::fs::write(&src, source).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg(flag)
        .current_dir(repo())
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

fn kernel_with(cond: &str) -> String {
    format!(
        "@require({cond})\nkernel k(Out: GlobalMemory<U32>) {{\n    let s = shared_alloc_u32(4);\n}}\nfn main() {{}}\n"
    )
}

// ---------------------------------------------------------------------------

/// An unsatisfiable requirement stops the build, at every backend, with the
/// documented code.
///
/// It is checked before backend dispatch on purpose - every emitter is reached
/// through that dispatch, so one check refuses uniformly rather than five.
#[test]
fn an_unsatisfiable_requirement_stops_every_backend() {
    let src = kernel_with("sm >= 9999");
    let mut checked = 0usize;
    for flag in [
        "--emit-ptx",
        "--emit-llvm",
        "--emit-cpu",
        "--emit-native",
        "--emit-coprocessor",
    ] {
        let (ok, text) = compile(&format!("uns{}", flag.trim_start_matches('-')), &src, flag);
        assert!(!ok, "{flag} accepted an unsatisfiable @require:\n{text}");
        assert!(
            text.contains("R0001") && text.contains("hardware requirement unsatisfied"),
            "{flag} refused, but not with the documented diagnostic:\n{text}"
        );
        checked += 1;
    }
    assert_eq!(checked, 5, "the backend sweep did not run");
}

/// The condition is EVALUATED, not answered by a constant.
///
/// Reads the host's own value out of the refusal message, then asserts the
/// requirement is satisfied at that value and unsatisfied one above it. A
/// compiler that always refuses fails the first half; one that always accepts
/// (the state this replaces) fails the second. Machine-independent by
/// construction.
#[test]
fn the_condition_is_actually_evaluated() {
    let (ok, text) = compile("probe", &kernel_with("sm >= 9999"), "--emit-ptx");
    assert!(!ok, "`sm >= 9999` was accepted:\n{text}");

    // "... but this host reports `sm = 89`."
    let marker = "reports `sm = ";
    let i = text
        .find(marker)
        .unwrap_or_else(|| panic!("the refusal does not report the host's value:\n{text}"));
    let rest = &text[i + marker.len()..];
    let have: i64 = rest[..rest.find('`').expect("unterminated value")]
        .parse()
        .expect("host value is not an integer");

    let (ok_at, t1) = compile("at", &kernel_with(&format!("sm >= {have}")), "--emit-ptx");
    assert!(ok_at, "`sm >= {have}` must be satisfied on a host reporting {have}:\n{t1}");

    let (ok_above, t2) =
        compile("above", &kernel_with(&format!("sm >= {}", have + 1)), "--emit-ptx");
    assert!(
        !ok_above,
        "`sm >= {}` must NOT be satisfied on a host reporting {have} - the boundary is \
         one unit wide and the gate is not reading it:\n{t2}",
        have + 1
    );
}

/// An unknown feature and an unevaluable shape are refused, not assumed
/// satisfied.
#[test]
fn what_it_cannot_answer_is_refused_rather_than_assumed() {
    for (tag, cond, code) in [
        ("unknown", "this_feature_does_not_exist >= 1", "R0002"),
        ("shape_lit", "1 == 0", "R0003"),
        ("shape_and", "avx512 >= 1 && sm >= 1", "R0003"),
    ] {
        let (ok, text) = compile(tag, &kernel_with(cond), "--emit-ptx");
        assert!(!ok, "`@require({cond})` was accepted:\n{text}");
        assert!(
            text.contains(code),
            "`@require({cond})` must be refused as {code}:\n{text}"
        );
    }
}

/// `@require` on anything but a `kernel` is refused rather than discarded.
///
/// Both forms were measured compiling cleanly with the requirement recorded
/// nowhere: a `fn` (§9.1's own example shape) and an `impl`
/// (`self_hosted/lib.ysu` had three, above `impl Vec`, `impl String` and
/// `impl File` - `@require(avx512 >= 1)` for `String::len`, which is what a
/// directive that does nothing looks like after a while).
#[test]
fn require_on_a_non_kernel_item_is_refused_not_discarded() {
    let cases = [
        (
            "fn",
            "@require(avx512 >= 1)\nfn f(a: I32) -> I32 { return a; }\nfn main() -> I32 { return f(1); }\n",
        ),
        (
            "impl",
            "struct V { x: I32 }\n@require(avx512 >= 1)\nimpl V {\n    fn g(v: I32) -> I32 { return v; }\n}\nfn main() -> I32 { return 0; }\n",
        ),
    ];
    for (tag, src) in cases {
        let (ok, text) = compile(&format!("nonk{tag}"), src, "--emit-llvm");
        assert!(!ok, "`@require` above a {tag} was accepted and discarded:\n{text}");
        assert!(
            text.contains("@require") && text.contains("kernel"),
            "refused above a {tag}, but not by name:\n{text}"
        );
    }
}

/// The control: a satisfiable requirement must still compile, and the
/// documented command that uses one must still work.
///
/// Without this, "refuse every `@require`" passes every assertion above while
/// deleting the feature - the `ordinary_loop_bodies_still_verify` shape.
/// `tests/test_drift.ysu` carries `@require(sm >= 89)` and is presented in
/// `CLAUDE.md` as a documented `--emit-ptx` invocation, so it is a real
/// end-to-end control rather than a fixture written to pass.
#[test]
fn a_satisfiable_requirement_still_compiles() {
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg("tests/test_drift.ysu")
        .arg("--emit-ptx")
        .current_dir(repo())
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "the documented `tests/test_drift.ysu --emit-ptx` must still compile; it carries \
         `@require(sm >= 89)`:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // And a kernel with no requirement at all is unaffected.
    let (ok, text) = compile(
        "norequire",
        "kernel k(Out: GlobalMemory<U32>) {\n    let s = shared_alloc_u32(4);\n}\nfn main() {}\n",
        "--emit-ptx",
    );
    assert!(ok, "a kernel with no `@require` must compile:\n{text}");
}

/// The advertised feature set, the evaluator and the manual must agree.
///
/// A name on `KNOWN_FEATURES` that the evaluator cannot resolve would be
/// refused with a message telling the user it is supported; a feature the
/// evaluator answers but the manual omits is undiscoverable. Asserting the two
/// AGREE rather than re-deriving a third copy of the table is the device this
/// repository already uses for the PTX `.version` floors.
#[test]
fn the_manual_lists_exactly_the_features_the_compiler_answers() {
    let doc = std::fs::read_to_string(repo().join("docs/y_language_documentation.md")).unwrap();
    let sec = doc
        .split("### 9.1 `@require`")
        .nth(1)
        .expect("§9.1 not found")
        .split("### 9.2")
        .next()
        .unwrap();

    let mut listed = 0usize;
    for f in y::require::KNOWN_FEATURES {
        assert!(
            sec.contains(&format!("`{f}`")),
            "`{f}` is answered by the compiler and is not listed in §9.1"
        );
        listed += 1;
    }
    assert!(listed >= 5, "only {listed} features checked against the manual");

    // The three diagnostics the manual names must exist in the compiler.
    let src = std::fs::read_to_string(repo().join("src/require.rs")).unwrap();
    for code in ["R0001", "R0002", "R0003", "R0004"] {
        assert!(
            src.contains(code),
            "the manual names {code} and `src/require.rs` does not emit it"
        );
        assert!(sec.contains(code), "{code} is emitted and §9.1 does not mention it");
    }
}

/// CPU features must be read from the LIVE authority, not from the cached
/// profile - and that can only be checked at the source.
///
/// On this machine `sentinel::host_has_avx512()` and
/// `HardwareProfile::has_avx512` agree, so swapping one for the other changes
/// no answer and no behavioural test can see it. That is the standing limit of
/// an agreement assertion, and this repository has already been bitten by it
/// once: a cached `.ysu_hw_profile` can be stale or copied from a better
/// machine, and reading AVX-512 from it is how `--emit-cpu` came to emit
/// AVX-512 dispatch on hardware that faults on it.
///
/// What separates the two readings is not their output but **which input each
/// consumes**, so it is pinned where the input is chosen. The predicates
/// themselves already carry the `XGETBV` check; this only asserts that
/// `require.rs` asks them.
#[test]
fn cpu_features_come_from_the_machine_not_the_cached_profile() {
    let src = std::fs::read_to_string(repo().join("src/require.rs")).unwrap();
    for p in ["host_has_avx()", "host_has_avx512()", "host_has_avx512_vnni()"] {
        assert!(
            src.contains(p),
            "`require.rs` must answer CPU features with `sentinel::{p}` - the live \
             reading - rather than from the cached profile"
        );
    }
    for bad in ["hw.has_avx512", "hw.has_avx", "profile.has_avx"] {
        assert!(
            !src.contains(bad),
            "`require.rs` reads `{bad}` from the cached hardware profile. A profile can \
             be stale or copied from a better machine; a wrong-high answer here is an \
             illegal instruction, not a slow kernel."
        );
    }
    // The GPU half legitimately DOES come from the profile - asserting that
    // too, so "read everything live" cannot pass either. There is no live GPU
    // probe on a CPU-only compile, and the profile is what selects the target.
    assert!(
        src.contains("hw.sm_version") && src.contains("hw.sm_count"),
        "GPU facts must come from the profile, which is what selects the PTX target"
    );
}

/// The claim that `sentinel.rs` resolves `@require` was false and must not
/// come back.
///
/// It is a source-level assertion because it is a claim about a DOCUMENT, and
/// running the compiler on a machine where the requirement happens to hold says
/// nothing about what §2 tells a reader. Same device as pinning which CPUID
/// reading a predicate uses.
#[test]
fn the_architecture_section_names_the_module_that_does_the_work() {
    let doc = std::fs::read_to_string(repo().join("docs/y_language_documentation.md")).unwrap();
    let sentinel = std::fs::read_to_string(repo().join("src/sentinel.rs")).unwrap();

    // The premise of the correction, re-measured every run rather than trusted.
    assert!(
        !sentinel.contains("@require"),
        "`sentinel.rs` handles `@require` now - §2 and this test both need updating"
    );
    // Anchored on the PROSE entry, which names the file, not on any line
    // mentioning the phrase: a Mermaid node in the same section reads
    // `E[Hardware Sentinel Resolver]`, and taking the first textual match
    // finds that instead. Caught by this test failing on its first run.
    let line = doc
        .lines()
        .find(|l| l.contains("**Hardware Sentinel Resolver (`sentinel.rs`)**"))
        .expect("the architecture entry for sentinel.rs is gone");
    assert!(
        !line.contains("Matches hardware constraints specified by `@require`"),
        "§2 credits `sentinel.rs` with resolving `@require`, which it does not do"
    );
    assert!(
        line.contains("require.rs"),
        "§2 should point at the module that actually evaluates the condition"
    );
}
