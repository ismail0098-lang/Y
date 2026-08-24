//! The LLVM backend targets the host. A GPU-only construct reaching it must be
//! refused by name, not transcribed.
//!
//! `cpu_emitter` already made this check - `tests/cpu_emitter_output_compiles.rs`
//! has `a_gpu_intrinsic_is_refused_rather_than_transcribed`. The LLVM backend
//! never got it, and the result was that **21 of the 76 corpus programs it
//! accepted produced an artifact that could not work**, all under
//! "Compilation Successful!" and exit 0:
//!
//! * **11 emitted invalid IR.** `emit_type`'s `Type::Ident` arm never consulted
//!   the struct table at all - it emitted `%Name` for anything that was not a
//!   primitive or an enum. `U32x4`, the PTX backend's 16-byte vector type,
//!   became a reference to an LLVM struct the module never defines, and
//!   `alloca %U32x4` on an undefined type is "Cannot allocate unsized type".
//!
//! * **10 referenced symbols that do not exist.** Any called name that was
//!   neither declared in the prelude nor defined in the module was
//!   auto-declared as `declare i32 @name(...)`. That module ASSEMBLES, and
//!   then fails at link with `undefined reference to 'thread_idx_x'` - which
//!   reads as a broken toolchain rather than as a program using a construct
//!   this backend cannot lower. Every such name in the corpus was a GPU
//!   intrinsic: `thread_idx_x`, `block_idx_x/y/z`, `bvh_traverse`,
//!   `rt_nearest_neighbor`, the carry-chain intrinsics, the v4 vector loads.
//!
//! The corpus sweep below is the load-bearing test, because it uses clang as
//! an oracle rather than asserting on the emitter's vocabulary - the lesson
//! from `feedback-decorative-codegen-passes-every-test`. It is paired with a
//! floor on how many programs still emit, so a backend that started refusing
//! everything would fail rather than pass vacuously.
use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn compile(name: &str, src: &str) -> (bool, String, PathBuf) {
    let dir = std::env::temp_dir().join(format!("y_llvmgpu_{}_{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{}.ysu", name));
    std::fs::write(&path, src).expect("write source");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .arg("--emit-llvm")
        .current_dir(repo())
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text, dir.join(format!("{}.ll", name)))
}

#[test]
fn a_gpu_intrinsic_is_refused_rather_than_declared() {
    let (ok, text, _) = compile(
        "gpuintr",
        "fn main() -> I32 {\n    let t: I32 = thread_idx_x();\n    return t;\n}\n",
        );
    assert!(!ok, "a GPU intrinsic compiled and exited 0:\n{}", text);
    assert!(
        text.contains("thread_idx_x"),
        "refused without naming the intrinsic:\n{}",
        text
    );
    assert!(
        text.contains("LLVM host backend"),
        "refused without saying which backend:\n{}",
        text
    );
}

#[test]
fn a_gpu_only_type_is_refused_rather_than_named() {
    let (ok, text, _) = compile(
        "gputype",
        "@unsafe\nfn f() {\n    let v: U32x4;\n}\n\nfn main() -> I32 {\n    f();\n    return 0;\n}\n",
    );
    assert!(!ok, "a GPU-only type compiled and exited 0:\n{}", text);
    assert!(
        text.contains("U32x4"),
        "refused without naming the type:\n{}",
        text
    );
}

/// The control. Refusing every unknown name would satisfy both tests above and
/// break the language - `String_new` is a real symbol in `c_src/runtime.c`, and
/// the first version of this refusal suppressed its declaration and turned two
/// valid modules into invalid ones.
///
/// It is verified NON-VACUOUS rather than assumed to be: the emitted module
/// must contain `declare ptr @String_new`, i.e. the fixture really does reach
/// the auto-declare path this refusal sits on. Dropping `String_new` from
/// `RUNTIME_SYMBOLS` fails this test and nothing else.
#[test]
fn ordinary_host_code_still_emits_valid_ir() {
    let (ok, text, ll) = compile(
        "hostok",
        "struct P {\n    a: I32,\n    b: I32,\n}\n\nfn main() -> I32 {\n    let p: P = P { a: 1, b: 5 };\n    let s = String_new(\"hi\");\n    println(&s);\n    return p.b;\n}\n",
    );
    assert!(ok, "ordinary host code was refused:\n{}", text);
    let ir = std::fs::read_to_string(&ll).expect("read emitted module");
    assert!(
        ir.contains("declare ptr @String_new"),
        "this fixture no longer reaches the auto-declare path, so it cannot \
         guard the refusal that sits on it:\n{}",
        ir
    );
    if Command::new("clang").arg("--version").output().is_err() {
        eprintln!("skipped clang check: no clang on this machine");
        return;
    }
    let out = Command::new("clang")
        .arg("-S")
        .arg("-o")
        .arg("/dev/null")
        .arg(&ll)
        .output()
        .expect("run clang");
    assert!(
        out.status.success(),
        "clang rejected the module for ordinary host code:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A struct used by a function declared BEFORE it must still resolve.
///
/// `emit_program`'s phase 0 used to register structs and resolve function
/// signatures in one loop, so `emit_type` was asked about a struct the table
/// did not have yet. That was invisible while the fallback silently emitted
/// `%Name` anyway - the moment it became a refusal, a forward reference became
/// a compile error. A name resolver must see all the names before it answers
/// any question.
///
/// This test exists because MUTATION found the hole: undoing the phase split
/// left every other test in this file green, and no program in the corpus
/// happens to declare a function above the struct it returns.
#[test]
fn a_struct_may_be_declared_after_the_function_that_uses_it() {
    let (ok, text, _) = compile(
        "fwdstruct",
        "fn make() -> P {\n    return P { a: 1, b: 5 };\n}\n\nstruct P {\n    a: I32,\n    b: I32,\n}\n\nfn main() -> I32 {\n    let p: P = make();\n    return p.b;\n}\n",
    );
    assert!(
        ok,
        "a struct declared below the function returning it was refused:\n{}",
        text
    );
}

/// The sweep. Every `.ysu` the backend ACCEPTS must produce a module clang
/// accepts - the property that was false for 11 of 76 programs.
///
/// It does NOT subsume `a_gpu_only_type_is_refused_rather_than_named`, and
/// mutation is what showed that: dropping the type refusal leaves this test
/// green, because every corpus program using `U32x4` also CALLS a GPU
/// intrinsic and is refused by the other check first. The working path masks
/// the broken one, so the narrow test earns its place.
#[test]
fn every_emitted_module_is_valid_llvm_ir() {
    if Command::new("clang").arg("--version").output().is_err() {
        eprintln!("skipped: no clang on this machine");
        return;
    }
    let tests_dir = repo().join("tests");
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&tests_dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "ysu").unwrap_or(false))
        .collect();
    sources.sort();
    assert!(sources.len() > 50, "corpus shrank to {} files", sources.len());

    let work = std::env::temp_dir().join(format!("y_llvmsweep_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();

    let mut emitted = 0usize;
    let mut refused = 0usize;
    let mut bad: Vec<String> = Vec::new();

    for src in &sources {
        let stem = src.file_stem().unwrap().to_string_lossy().to_string();
        let copy = work.join(format!("{}.ysu", stem));
        std::fs::copy(src, &copy).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&copy)
            .arg("--emit-llvm")
            .current_dir(repo())
            .output()
            .expect("run Y");
        let ll = work.join(format!("{}.ll", stem));
        if !out.status.success() || !ll.exists() {
            refused += 1;
            continue;
        }
        emitted += 1;
        let c = Command::new("clang")
            .arg("-S")
            .arg("-o")
            .arg("/dev/null")
            .arg(&ll)
            .output()
            .expect("run clang");
        if !c.status.success() {
            let err = String::from_utf8_lossy(&c.stderr);
            let first = err
                .lines()
                .find(|l| l.contains("error"))
                .unwrap_or("(no error line)");
            bad.push(format!("  {}: {}", stem, first));
        }
    }
    let _ = std::fs::remove_dir_all(&work);

    // Without this, a backend that refused everything would pass perfectly -
    // the shape `the_corpus_is_not_all_skips` exists for in the cross-backend
    // differential.
    assert!(
        emitted >= 40,
        "only {} of {} programs emitted a module - the backend is refusing \
         almost everything, so this test proves nothing",
        emitted,
        sources.len()
    );
    assert!(
        bad.is_empty(),
        "{} of {} emitted modules are invalid LLVM IR (refused: {}):\n{}",
        bad.len(),
        emitted,
        refused,
        bad.join("\n")
    );
}

/// The emitter's list of runtime symbols must agree with the runtime, rather
/// than be a second copy of it that drifts. Asserting AGREEMENT is the device
/// the `.version` gate uses; re-deriving the table in the test would just be a
/// third copy.
///
/// A name in this list that the runtime does not define is the dangerous
/// direction: it licenses a `declare` for a symbol the link cannot resolve,
/// which is exactly the failure this whole file exists to stop.
#[test]
fn runtime_symbols_match_the_runtime() {
    let sources = runtime_sources();

    let missing: Vec<&str> = y::llvm_emitter::RUNTIME_SYMBOLS
        .iter()
        .copied()
        .filter(|name| find_definitions(&sources, name).is_empty())
        .collect();
    assert!(
        missing.is_empty(),
        "the LLVM emitter allows these symbols but the runtime does not \
         define them, so a call to one would be declared and then fail to \
         link: {:?}",
        missing
    );

    // A `static` definition emits NO SYMBOL, so it satisfies "is defined" and
    // still fails to link. That is not hypothetical: the whole ShadowPlay GUI
    // surface was changed to `static inline` to silence a warning, and the
    // only application in the repo that calls it stopped linking with
    // `undefined reference to 'init_shadowplay_gui'`. The old version of this
    // test looked for a definition and passed the entire time.
    let unlinkable: Vec<&str> = y::llvm_emitter::RUNTIME_SYMBOLS
        .iter()
        .copied()
        // EVERY definition, not the first one found. `shadowplay_gui.h` has
        // two - a real one and a `Y_NO_X11` stub - and only one is compiled.
        // Checking the first meant the stub branch, which sits earlier in the
        // file, masked a re-`static`d real branch: verified by mutation, this
        // test passed with the exact original bug put back.
        .filter(|name| {
            let defs = find_definitions(&sources, name);
            !defs.is_empty() && defs.iter().any(|d| d.is_static)
        })
        .collect();
    assert!(
        unlinkable.is_empty(),
        "these symbols are defined `static`, so no symbol is emitted and a \
         call to one links to nothing: {:?}",
        unlinkable
    );
}

struct Definition {
    is_static: bool,
}

/// `c_src/runtime.c` and every repo header it includes. A symbol the LLVM path
/// links against does not have to be written in the `.c` file - the ShadowPlay
/// surface lives in `shadowplay_gui.h`, and reading only `runtime.c` is what
/// let that list drift.
fn runtime_sources() -> Vec<String> {
    let mut out = Vec::new();
    let c = std::fs::read_to_string(repo().join("c_src/runtime.c")).expect("read c_src/runtime.c");
    for line in c.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("#include \"") {
            if let Some(name) = rest.split('"').next() {
                let path = repo().join("c_src").join(name);
                if let Ok(text) = std::fs::read_to_string(&path) {
                    out.push(text);
                }
            }
        }
    }
    out.push(c);
    out
}

/// A DEFINITION, not a call or a forward declaration: the name at a word
/// boundary, followed by a parameter list, followed by `{`. A call is followed
/// by `;`, and so is a prototype - both were counted as definitions before.
fn find_definitions(sources: &[String], name: &str) -> Vec<Definition> {
    let mut found = Vec::new();
    for src in sources {
        for (i, _) in src.match_indices(name) {
            let before_ok = i == 0 || {
                let b = src.as_bytes()[i - 1];
                !b.is_ascii_alphanumeric() && b != b'_'
            };
            if !before_ok {
                continue;
            }
            let after = &src[i + name.len()..];
            let after = after.trim_start();
            if !after.starts_with('(') {
                continue;
            }
            let close = match after.find(')') {
                Some(c) => c,
                None => continue,
            };
            if !after[close + 1..].trim_start().starts_with('{') {
                continue; // a call or a prototype
            }
            let line_start = src[..i].rfind('\n').map(|n| n + 1).unwrap_or(0);
            let prefix = &src[line_start..i];
            found.push(Definition {
                is_static: prefix.split_whitespace().any(|w| w == "static"),
            });
        }
    }
    found
}
