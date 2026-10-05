//! `@cache_policy` is lowered by `--emit-ptx` and refused by every other
//! backend; `@gpu_uncached` is `volatile` and no longer non-temporal;
//! `@prefetch_stride` is refused, because nothing lowers it.
//!
//! Measured before any of this was written, x86-64 at `clang -O2` (the flags
//! the LLVM backend's own link step uses):
//!
//! * **`@cache_policy` on the LLVM backend did nothing its four policies name,
//!   and every one exited 0.** `L2_EVICT_FIRST` put `!nontemporal` on a scalar
//!   load, which compiled to the same `movq (%rdi), %rax` as no policy.
//!   `L2_PERSIST` emitted `llvm.prefetch` of the address being loaded, which
//!   came out as a `prefetcht0` scheduled AFTER the load of that address.
//!   `L2_EVICT_LAST` and `L2_STREAM` were dropped. The policies are NVIDIA L2
//!   eviction priorities, and x86 has no instruction that sets a cache line's
//!   eviction priority.
//! * **Three backends accepted a policy they never read.** Only `ptx_emitter`
//!   and `llvm_emitter` mention `cache_policy`; `--emit-cpu`, `--emit-native`
//!   and the LLVM backend all compiled a policy on a `let` that loads nothing
//!   and exited 0, and the ZK and co-processor backends read no `let` attribute
//!   at all. `--emit-cpu` is also what `c_api::y_interpret_kernel` runs, which
//!   is why the refusal lives in each emitter rather than only in `main.rs`.
//! * **`@gpu_uncached` stored with `store volatile ... !nontemporal`, which
//!   x86-64 lowers to `movnti`.** A non-temporal store is the one kind of
//!   ordinary store the x86 memory model allows to become visible before an
//!   earlier store, so `c.data = v; c.status = 1;` could publish the flag ahead
//!   of the data it guards - the status-flag use §9.5 advertised the attribute
//!   for, under "guarantee immediate visibility". On the load side
//!   `!nontemporal` compiled to the same `mov`.
//! * **`@prefetch_stride` had one reader**: the LLVM backend, which wrote it
//!   into the module as a comment ("solver-guided cache warming") and emitted no
//!   prefetch. The PTX backend never read it. A reader is not proof the reader
//!   does anything.
//!
//! No program in `tests/` uses any of the three directives, which is how all
//! of this survived: nothing ever compiled them.
//!
//! The census asks the compiler for its own list of backend flags, so a
//! backend added later is checked without anyone remembering to come back here
//! - and a flag this file does not know how to check FAILS rather than being
//! skipped.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use y::lexer::Lexer;
use y::parser::Parser;

#[path = "common/pinned.rs"]
mod pinned;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// A fresh directory per call. The tag is for reading a leftover directory;
/// the counter is what makes two calls with one tag distinct.
fn scratch(tag: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let d = std::env::temp_dir().join(format!(
        "y_host_cache_{tag}_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("create scratch dir");
    d
}

/// Runs the compiler in `dir`, against a pinned hardware profile, on `src`
/// written there. It used to run from the repository root "so the hardware
/// profile and the runtime are the ones every other test uses" - but the
/// runtime is found from the compiler's own path, and the repository's profile
/// is this machine's card. Every artifact goes to `dir`: `-o` is always given,
/// because `--emit-native` otherwise writes `output_bin` into the working
/// directory.
fn compile(dir: &PathBuf, name: &str, src: &str, flag: &str) -> (bool, String) {
    let path = dir.join(format!("{name}.ysu"));
    std::fs::write(&path, src).expect("write source");
    pinned::pin(dir, pinned::SM_PINNED);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(&path).current_dir(dir);
    if !flag.is_empty() {
        cmd.arg(flag);
    }
    cmd.arg("-o").arg(dir.join(format!("{name}.out")));
    let out = cmd.output().expect("run Y");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

/// Scalar code every non-GPU backend accepts, circuits included: the ZK
/// backend needs an input, or the output is a constant and the circuit has no
/// constraint to prove.
const SCALAR_CTL: &str = "fn main(a: I32) -> I32 {\n    let t: I32 = a * a;\n    let v: I32 = t;\n    return v;\n}\n";
const SCALAR_POLICY: &str = "fn main(a: I32) -> I32 {\n    let t: I32 = a * a;\n    @cache_policy(L2_STREAM)\n    let v: I32 = t;\n    return v;\n}\n";

/// The one backend that lowers the directive: `L2_STREAM` is `ld.global.cs`.
const PTX_POLICY: &str = "kernel k(Src: GlobalMemory<F32>, Out: GlobalMemory<F32>) {\n    @cache_policy(L2_STREAM)\n    let v: F32 = Src[1];\n    Out[0] = v;\n}\nfn main() {}\n";

fn coprocessor_program(with_policy: bool) -> String {
    let src = std::fs::read_to_string(repo().join("tests/coprocessor_collision.ysu"))
        .expect("read tests/coprocessor_collision.ysu");
    let anchor = "    let collision_contacts: F32";
    assert_eq!(src.matches(anchor).count(), 1, "the co-processor fixture moved");
    if with_policy {
        src.replace(anchor, &format!("    @cache_policy(L2_STREAM)\n{anchor}"))
    } else {
        src
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Class {
    /// Compiles plain scalar code: must refuse the policy, and compile the
    /// control, so a refusal cannot be for an unrelated reason.
    ScalarCode,
    /// Lowers the policy.
    Ptx,
    /// Compiles only RT/Tensor co-processor programs.
    Coprocessor,
    /// A removed backend: refuses every program.
    Removed,
    /// Does not compile a `.ysu` source at all.
    NoSource,
    /// Runs the front end and generates no code (`--emit-guarantees`, which
    /// writes what the type checker established): a policy has nothing to be
    /// lowered into, so it is neither honoured nor refused. Checked to accept
    /// the program and write its table, not code.
    NoCode,
}

fn classify(flag: &str) -> Option<Class> {
    match flag {
        "" | "--emit-llvm" | "--target=llvm" | "--emit-cpu" | "--target=cpu" | "--emit-native"
        | "--target=native" | "--emit-r1cs" | "--target=r1cs" | "--emit-zk-ptx"
        | "--target=zk-ptx" => Some(Class::ScalarCode),
        "--emit-ptx" | "--target=ptx" => Some(Class::Ptx),
        "--emit-coprocessor" | "--target=coprocessor" => Some(Class::Coprocessor),
        "--emit-c" | "--target=c" | "--c" => Some(Class::Removed),
        // `--emit-attention-ptx` generates the exact-attention kernel from its
        // own parameters, and `--emit-verifier` reads a snarkjs key: neither
        // compiles the source program a `@cache_policy` would be written in.
        "--emit-attention-ptx" | "--emit-verifier" => Some(Class::NoSource),
        "--emit-guarantees" => Some(Class::NoCode),
        _ => None,
    }
}

/// The backend-selecting flags, from the CLI's own `Known options:` line.
fn backend_flags() -> Vec<String> {
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(repo().join("tests/hello.ysu"))
        .arg("--this-is-not-a-flag")
        .current_dir(repo())
        .output()
        .expect("run Y");
    let text = String::from_utf8_lossy(&out.stdout).into_owned()
        + &String::from_utf8_lossy(&out.stderr);
    let line = text
        .lines()
        .find(|l| l.contains("Known options:"))
        .expect("the CLI lists its known options for an unrecognised flag");
    let flags: Vec<String> = line
        .split_whitespace()
        .filter(|t| t.starts_with("--emit-") || t.starts_with("--target=") || *t == "--c")
        .map(str::to_string)
        .collect();
    assert!(
        flags.len() >= 10,
        "the backend-flag list looks truncated: {line}"
    );
    flags
}

fn names_the_refusal(text: &str) -> bool {
    text.contains("@cache_policy(L2_STREAM)") && text.contains("has no lowering on this target")
}

/// **The census: every backend either lowers `@cache_policy` or refuses it by
/// name.** Exiting 0 with the directive dropped is the one thing none may do.
#[test]
fn every_backend_honours_or_refuses_a_cache_policy() {
    let mut flags = vec![String::new()]; // no flag: the default LLVM backend
    flags.extend(backend_flags());

    let unclassified: Vec<&String> = flags.iter().filter(|f| classify(f).is_none()).collect();
    assert!(
        unclassified.is_empty(),
        "the CLI knows backend flags this census does not classify: {unclassified:?}. \
         Decide whether each lowers `@cache_policy` or refuses it, and add it to `classify`."
    );

    let mut refused = 0usize;
    let mut honoured = 0usize;
    let mut skipped = Vec::new();
    for flag in &flags {
        let shown = if flag.is_empty() { "(default)" } else { flag.as_str() };
        let d = scratch("census");
        match classify(flag).unwrap() {
            Class::ScalarCode => {
                let (ok, out) = compile(&d, "ctl", SCALAR_CTL, flag);
                if !ok && out.contains("not compiled into this binary") {
                    skipped.push(format!("{shown}: backend not in this build"));
                    continue;
                }
                assert!(ok, "{shown} must compile the control program, or a refusal below says nothing:\n{out}");
                let (ok, out) = compile(&d, "pol", SCALAR_POLICY, flag);
                assert!(
                    !ok && names_the_refusal(&out),
                    "{shown} accepted `@cache_policy` or refused it without naming it. It has no \
                     cache-policy lowering, so it must refuse the directive by name:\n{out}"
                );
                refused += 1;
            }
            Class::Coprocessor => {
                let (ok, out) = compile(&d, "ctl", &coprocessor_program(false), flag);
                assert!(ok, "{shown} must compile the control program:\n{out}");
                let (ok, out) = compile(&d, "pol", &coprocessor_program(true), flag);
                assert!(
                    !ok && names_the_refusal(&out),
                    "{shown} accepted `@cache_policy` or refused it without naming it:\n{out}"
                );
                refused += 1;
            }
            Class::Ptx => {
                let (ok, out) = compile(&d, "pol", PTX_POLICY, flag);
                assert!(ok, "{shown} lowers `@cache_policy` and must compile this kernel:\n{out}");
                // `-o` names the artifact for this backend.
                let ptx = std::fs::read_to_string(d.join("pol.out"))
                    .or_else(|_| std::fs::read_to_string(d.join("pol.ptx")))
                    .expect("read the emitted PTX");
                assert!(
                    ptx.contains("ld.global.cs.f32"),
                    "{shown} compiled the kernel but `L2_STREAM` is not in the load:\n{ptx}"
                );
                honoured += 1;
            }
            Class::Removed => {
                let (ok, _) = compile(&d, "ctl", SCALAR_CTL, flag);
                assert!(!ok, "{shown} is a removed backend and must still refuse every program");
            }
            Class::NoSource => skipped.push(format!("{shown}: compiles no source program")),
            Class::NoCode => {
                let (ok, out) = compile(&d, "pol", SCALAR_POLICY, flag);
                assert!(ok, "{shown} writes the front end's table and must accept the program:\n{out}");
                let table = std::fs::read_to_string(d.join("pol.out")).expect("read the table");
                assert!(table.starts_with("{\"version\": 1"), "{shown} wrote something other than its table:\n{table}");
            }
        }
        let _ = std::fs::remove_dir_all(&d);
    }
    eprintln!("census: {refused} refused, {honoured} honoured; skipped: {skipped:?}");
    // Non-vacuity: the default LLVM backend, `--emit-llvm`, `--target=llvm`,
    // the two CPU and two native spellings and the two co-processor spellings
    // are present in every build.
    assert!(refused >= 9, "only {refused} backends were checked for a refusal");
    assert!(honoured >= 2, "the PTX backend was not checked for the lowering");
}

/// The query every backend refuses through must find a policy wherever a `let`
/// can be written: in each kind of item, and in each kind of nested block.
#[test]
fn a_policy_is_found_in_every_place_a_let_can_be_written() {
    let src = r#"
struct Holder { x: I32 }
impl Holder {
    fn get(h: &Holder) -> I32 {
        @cache_policy(L2_STREAM)
        let a: I32 = 1;
        return a;
    }
}
module Inner {
    fn deep() -> I32 {
        @cache_policy(L2_STREAM)
        let b: I32 = 2;
        return b;
    }
}
kernel k(Out: GlobalMemory<F32>) {
    @cache_policy(L2_STREAM)
    let c: F32 = Out[0];
    Out[1] = c;
}
fn main() -> I32 {
    let mut n: I32 = 0;
    if n == 0 {
        @cache_policy(L2_STREAM)
        let d: I32 = 3;
        n = d;
    } else {
        @cache_policy(L2_STREAM)
        let e: I32 = 4;
        n = e;
    }
    @invariant(n >= 0)
    while n < 10 {
        @cache_policy(L2_STREAM)
        let f: I32 = 1;
        n = n + f;
    }
    @invariant(i >= 0)
    for i in 0..2 {
        @cache_policy(L2_STREAM)
        let g: I32 = i;
        n = n + g;
    }
    @ghost {
        @cache_policy(L2_STREAM)
        let h: I32 = 5;
        n = n + h;
    }
    return n;
}
"#;
    let tokens = Lexer::new(src).tokenize();
    let program = Parser::new(tokens).parse_program().expect("parse");
    let mut found: Vec<String> = y::ast::cache_policy_sites(&program)
        .into_iter()
        .map(|s| s.binding)
        .collect();
    found.sort();
    assert_eq!(
        found,
        ["a", "b", "c", "d", "e", "f", "g", "h"],
        "`cache_policy_sites` missed a place a `let` can be written - a policy there would be \
         dropped by every backend that refuses through this query"
    );

    // And the refusal reaches a nested one through the real compiler. A
    // smaller program than the one above: that one's loops are for the parser,
    // and the invariant checker (rightly) refuses them before any backend runs.
    let nested = "fn main() -> I32 {\n    let mut n: I32 = 0;\n    if n == 0 {\n        @cache_policy(L2_STREAM)\n        let d: I32 = 3;\n        n = d;\n    }\n    return n;\n}\n";
    let d = scratch("nested");
    let (ok, out) = compile(&d, "nested", nested, "--emit-llvm");
    assert!(!ok && names_the_refusal(&out), "the LLVM backend accepted a nested policy:\n{out}");
    let control = nested.replace("        @cache_policy(L2_STREAM)\n", "");
    let (ok, out) = compile(&d, "nested_ctl", &control, "--emit-llvm");
    assert!(ok, "the same program without the policy must compile:\n{out}");
    let _ = std::fs::remove_dir_all(&d);
}

const UNCACHED: &str = "struct Chan {\n    data: I32,\n    @gpu_uncached status: I32,\n}\nfn publish(c: &mut Chan, v: I32) {\n    c.data = v;\n    c.status = 1;\n}\nfn poll(c: &mut Chan) -> I32 {\n    return c.status;\n}\nfn main() {\n    print_int(1);\n}\n";

fn have_clang() -> bool {
    Command::new("clang").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// **`@gpu_uncached` is `volatile` and never non-temporal.** A `!nontemporal`
/// store is `movnti` on x86-64, which may become visible before an earlier
/// store - the status flag ahead of its data.
#[test]
fn gpu_uncached_is_volatile_and_never_non_temporal() {
    let d = scratch("uncached");
    let (ok, out) = compile(&d, "unc", UNCACHED, "--emit-llvm");
    assert!(ok, "the LLVM backend must compile the @gpu_uncached program:\n{out}");
    // On `--emit-llvm`, `-o` names the IR file itself; nothing is linked.
    let ll = std::fs::read_to_string(d.join("unc.out")).expect("read the emitted IR");

    assert!(
        !ll.contains("!nontemporal"),
        "the module carries `!nontemporal`; on x86-64 a non-temporal store is `movnti`, which \
         may be reordered ahead of an earlier store:\n{ll}"
    );
    // The flag is still volatile - the part of the attribute that is right -
    // and the ordinary field is not, so the attribute still does something.
    let flag_store = ll
        .lines()
        .find(|l| l.contains("store") && l.contains(" 1, ptr"))
        .expect("the flag store `c.status = 1`");
    assert!(flag_store.contains("store volatile i32"), "the flag store lost `volatile`: {flag_store}");
    let data_store = ll
        .lines()
        .find(|l| l.trim_start().starts_with("store") && l.contains("%v"))
        .or_else(|| ll.lines().find(|l| l.trim_start().starts_with("store i32") && !l.contains(" 1, ptr")));
    if let Some(s) = data_store {
        assert!(!s.contains("volatile"), "the ordinary field became volatile: {s}");
    }
    assert!(
        ll.lines().any(|l| l.contains("load volatile i32")),
        "the flag load lost `volatile`, so a polling loop could be hoisted:\n{ll}"
    );

    // What the processor runs: no non-temporal store anywhere.
    if have_clang() {
        let asm = d.join("unc.s");
        let st = Command::new("clang")
            .args(["-O2", "-S", "-Wno-override-module", "-x", "ir", "-o"])
            .arg(&asm)
            .arg(d.join("unc.out"))
            .status()
            .expect("run clang");
        assert!(st.success(), "clang rejected the emitted IR");
        let s = std::fs::read_to_string(&asm).expect("read assembly");
        assert!(!s.contains("movnti"), "the assembly still has a non-temporal store:\n{s}");
    } else {
        eprintln!("SKIP asm half: clang unavailable");
    }
    let _ = std::fs::remove_dir_all(&d);
}

/// **`@prefetch_stride` is refused by every backend**, and the same loop
/// without it compiles, so the refusal is about the directive.
#[test]
fn prefetch_stride_is_refused_on_every_backend() {
    let host = "fn main() -> I32 {\n    let mut acc: I32 = 0;\n    @prefetch_stride(64)\n    @invariant(i >= 0)\n    for i in 0..4 {\n        acc = acc + i;\n    }\n    return acc;\n}\n";
    // The form the language reference used: above a `let`, where the parser
    // used to drop it outright.
    let above_let = "fn main() -> I32 {\n    @prefetch_stride(64)\n    let v: I32 = 7;\n    return v;\n}\n";
    let kernel = "kernel k(Out: GlobalMemory<F32>, N: I32) {\n    @prefetch_stride(64)\n    @invariant(i >= 0)\n    for i in 0..N {\n        Out[i] = 1.0;\n    }\n}\nfn main() {}\n";
    for (src, flag) in [
        (host, "--emit-llvm"),
        (host, "--emit-cpu"),
        (host, ""),
        (above_let, "--emit-llvm"),
        (above_let, "--emit-native"),
        (kernel, "--emit-ptx"),
    ] {
        let d = scratch("prefetch");
        let (ok, out) = compile(&d, "pf", src, flag);
        assert!(
            !ok && out.contains("`@prefetch_stride` is not implemented"),
            "{flag:?} accepted `@prefetch_stride`, which no backend lowers:\n{out}"
        );
        let control = src.replace("    @prefetch_stride(64)\n", "");
        assert_ne!(control, src, "the control did not remove the directive");
        let (ok, out) = compile(&d, "ctl", &control, flag);
        assert!(ok, "{flag:?} must compile the same loop without the directive:\n{out}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
