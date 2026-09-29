//! A built-in call with the wrong number of arguments is refused, in BOTH
//! directions, naming the function.
//!
//! The PTX emitter had 62 `args.len() >= N` guards, six exact-count checks,
//! and no maximum anywhere, so an over-long call was silently TRUNCATED.
//! `store` is `store(place, value)`, and its natural spelling
//!
//! ```text
//! store(Out, 0, 7);        // buffer, index, value - like atomic_add
//! ```
//!
//! stored the INDEX: the card read back `00 00 00 00`, the compiler exited 0,
//! `ptxas` accepted it. Every test in this repository that used `store` had
//! written the three-argument form.
//!
//! Writing the maximum found two rows whose MINIMUM was wrong as well:
//! `block_ptr3d_store`'s stored value is argument 10 and fell back to the
//! register `%f0` - another variable - below that, while its minimum said 4;
//! and `block_ptr3d_store_v4` read its four values only at exactly 13
//! arguments, so 11 or 12 dropped the extras.
//!
//! The sweeps drive `PtxEmitter` directly, so a type-checker refusal cannot
//! stand in for the gate being tested; the end-to-end cases go through the
//! binary and the card. The census that every lowered name HAS a row is the
//! `tests_builtin_arity` unit module in `src/ptx_emitter.rs`.

use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

const HEADER: &str = "kernel k(A: GlobalMemory<F32>, B: GlobalMemory<F32>, N: I32)";

/// The PTX backend alone: (refusals, PTX if there were none).
fn backend(body: &str) -> (String, Option<String>) {
    let source = format!("{} {{\n    {}\n}}\n\nfn main() {{\n}}\n", HEADER, body);
    let program = y::parser::Parser::new(y::lexer::Lexer::new(&source).tokenize())
        .parse_program()
        .unwrap_or_else(|e| panic!("fixture does not parse: {:?}\n{}", e, source));
    let hardware = y::sentinel::HardwareProfile {
        sm_version: "8.9".into(),
        ..Default::default()
    };
    let mut emitter = y::ptx_emitter::PtxEmitter::new_with_profile(&hardware);
    let ptx = emitter.emit_program(&program, &hardware);
    let log = emitter.emit_errors.join("\n");
    (log, emitter.emit_errors.is_empty().then_some(ptx))
}

/// The real binary, on a copy in its own directory.
fn binary(src: &str) -> (bool, String, Option<String>) {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!(
        "y_builtin_arity_{}_{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("k.ysu");
    std::fs::write(&file, format!("{}\nfn main() {{}}\n", src)).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&file)
        .arg("--emit-ptx")
        .current_dir(repo)
        .output()
        .expect("run Y");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read_to_string(dir.join("k.ptx")).ok();
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.success(), log, ptx)
}

fn args(first: &str, n: usize) -> String {
    let mut v = vec![first.to_string()];
    v.extend(std::iter::repeat("N".to_string()).take(n.saturating_sub(1)));
    if n == 0 {
        return String::new();
    }
    v.join(", ")
}

/// Every name the `Expr::Call` lowering compares a callee against, recovered
/// from the emitter's source - the same list the census unit test checks has
/// a row, here driven through the gate itself.
fn lowered_names() -> Vec<String> {
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ptx_emitter.rs"))
        .unwrap();
    let mut out = Vec::new();
    for needle in ["fname == \"", "\n        \""] {
        let mut rest = src.as_str();
        while let Some(i) = rest.find(needle) {
            rest = &rest[i + needle.len()..];
            if let Some(end) = rest.find('"') {
                let name = &rest[..end];
                // The second needle is `carry_spec`'s row shape; keep only
                // its intrinsic names.
                let carry = needle.starts_with('\n')
                    && rest[end..].starts_with("\" => (\"")
                    && name.ends_with("_u32");
                if needle.starts_with('f') || carry {
                    out.push(name.to_string());
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// The Hopper / fragment surface is refused at ANY count by its own message,
/// which names the real reason - so it is not asked for an arity message.
const REFUSED_WHATEVER: &[&str] = &[
    "cp_async_bulk", "tma_load", "tma_load_2d", "wgmma_async", "wgmma_mma_async",
    "mbarrier_init", "mbarrier_arrive", "mbarrier_try_wait", "mma_sync",
];

/// Twenty arguments is more than any built-in takes (the most is 13). Every
/// lowered name must refuse it by name, saying how many it takes and that
/// it was given 20 - the maximum direction had no gate at all.
#[test]
fn every_built_in_refuses_too_many_arguments() {
    let names = lowered_names();
    assert!(names.len() >= 60, "recovered only {} names: {:?}", names.len(), names);
    let mut failures = Vec::new();
    for name in &names {
        let (log, ptx) = backend(&format!("{}({});", name, args("A", 20)));
        let named = log.contains(&format!("`{}(...)`", name));
        let counted = REFUSED_WHATEVER.contains(&name.as_str()) || log.contains("was given 20");
        if ptx.is_some() || !named || !counted {
            failures.push(format!("{}:\n{}", name, log));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} built-ins accepted or mis-reported 20 arguments:\n\n{}",
        failures.len(),
        names.len(),
        failures.join("\n\n")
    );
}

/// The same sweep for `Namespace::member` callees and pipeline methods.
#[test]
fn paths_and_methods_refuse_too_many_arguments() {
    for (call, name) in [
        (format!("barrier::sync({});", args("N", 1)), "barrier::sync"),
        (format!("BlockTile::load({});", args("A", 20)), "BlockTile::load"),
        (format!("BlockTile::store({});", args("A", 20)), "BlockTile::store"),
        (format!("let v: F32 = GlobalMemory::load({});", args("A", 2)), "GlobalMemory::load"),
        (format!("let v: F32 = GlobalMemory::load_v4({});", args("A", 2)), "GlobalMemory::load_v4"),
        (format!("let v: F32 = GlobalMemory::ld_v4({});", args("A", 2)), "GlobalMemory::ld_v4"),
        (format!("GlobalMemory::store_v4({});", args("A", 3)), "GlobalMemory::store_v4"),
        (format!("GlobalMemory::st_v4({});", args("A", 4)), "GlobalMemory::st_v4"),
        (
            "let tok: AsyncToken = cp_async(A, B, 16);\n    pipe.wait(tok, N);".to_string(),
            "<pipeline>.wait",
        ),
        ("pipe.commit(N);".to_string(), "<pipeline>.commit"),
    ] {
        let (log, ptx) = backend(&call);
        assert!(ptx.is_none(), "`{}` was accepted:\n{}", call, log);
        assert!(
            log.contains(&format!("`{}(...)`", name)) && log.contains("was given"),
            "`{}` was not refused with an arity message naming `{}`:\n{}",
            call,
            name,
            log
        );
    }
}

/// The rows whose count was wrong or partial, each at the count it used to
/// accept, with the reason the old behaviour was a wrong answer.
#[test]
fn the_rows_that_accepted_a_wrong_count_now_refuse_it() {
    for (call, takes) in [
        // Stored the INDEX.
        ("store(A, 0, 7);".to_string(), "exactly 2 arguments"),
        // Read its value from `%f0`, another variable's register.
        ("block_ptr3d_store(A, N, N, N);".to_string(), "exactly 10 arguments"),
        // 11 or 12: dropped the extras and broadcast the first value.
        (format!("block_ptr3d_store_v4({});", args("A", 11)), "10 or 13 arguments"),
        (format!("block_ptr3d_store_v4({});", args("A", 12)), "10 or 13 arguments"),
        // Two or three values: the missing lanes silently got the first.
        ("store_v4(A, N, N);".to_string(), "2 or 5 arguments"),
        ("GlobalMemory::store_v4(A, N, N, N);".to_string(), "2 or 5 arguments"),
        // A 4th argument turned unrolling on by its PRESENCE; its value was
        // discarded, so `vec_add_v4(A, B, A, 0)` unrolled too.
        ("vec_add_v4(A, B, A, 0);".to_string(), "exactly 3 arguments"),
        // Discarded: a nullary built-in given an argument.
        ("let t: I32 = thread_idx_x(5);".to_string(), "no arguments"),
        // Used to fall through to "no PTX lowering exists for this name -
        // check the spelling", which sends the reader looking for a typo.
        ("let w: U64 = mul_wide_u32(N, N, N);".to_string(), "exactly 2 arguments"),
        ("let c: U32 = add_cc_u32(N, N, N);".to_string(), "exactly 2 arguments"),
    ] {
        let (log, ptx) = backend(&call);
        assert!(ptx.is_none(), "`{}` was accepted:\n{}", call, log);
        assert!(log.contains(takes), "`{}` must say it takes {}:\n{}", call, takes, log);
    }
    let (log, _) = backend("store(A, 0, 7);");
    assert!(
        log.contains("store(Out[i], v)"),
        "the `store` refusal must name the spelling that works:\n{}",
        log
    );
    let (log, _) = backend("vec_add_v4(A, B, A, 0);");
    assert!(log.contains("vec_add_unrolled4"), "must name the unrolled form:\n{}", log);
}

/// The control: every accepted count of the same rows still compiles, so the
/// gate is on the count and not on the built-in.
#[test]
fn the_counts_each_built_in_takes_still_compile() {
    for call in [
        "store(A[0], 1.0);",
        "store(A, 1.0);",
        "block_ptr3d_store(A, 0, 0, 1, 1, 1, 1, 1, 4, 2.5);",
        "block_ptr3d_store_v4(A, 0, 0, 0, 1, 1, 1, 1, 4, 2.5);",
        "block_ptr3d_store_v4(A, 0, 0, 0, 1, 1, 1, 1, 4, 1.0, 2.0, 3.0, 4.0);",
        "store_v4(A, 1.0);",
        "store_v4(A, 1.0, 2.0, 3.0, 4.0);",
        "vec_add_v4(A, B, A);",
        "vec_add_unrolled4(A, B, A);",
        "let t: I32 = thread_idx_x();",
        "let w: U64 = mul_wide_u32(N, N);",
        "let tok: AsyncToken = cp_async(A, B, 16);\n    pipe.wait(tok);",
        "let tok: AsyncToken = cp_async(A, B);\n    pipe.wait(tok);",
        "let v: F32 = block_ptr2d_load(A, 0, N);",
        "let v: F32 = block_ptr2d_load(A, 0, N, N, 1, N);",
    ] {
        let (log, ptx) = backend(call);
        assert!(ptx.is_some(), "`{}` takes this many arguments and was refused:\n{}", call, log);
    }
}

/// `vec_add_v4` no longer unrolls on a 4th argument - by NAME only - so the
/// 3-argument form is still the single-iteration one, and the unrolled name
/// still unrolls. (With the 4-argument form refused this is what stops the
/// flag being silently re-derived from something else.)
#[test]
fn vec_add_unrolls_by_name_only() {
    let count = |ptx: &str| ptx.matches("add.f32").count();
    let one = backend("vec_add_v4(A, B, A);").1.expect("vec_add_v4 compiles");
    let four = backend("vec_add_unrolled4(A, B, A);").1.expect("vec_add_unrolled4 compiles");
    assert_eq!(count(&four), 4 * count(&one), "unrolled4 must be four copies of the body");
}

// ───────────────────────────── end to end ─────────────────────────────

/// The reported case through the real binary: refused, no artifact, exit
/// non-zero. It used to exit 0 and store the index.
#[test]
fn the_three_argument_store_fails_the_build() {
    let (ok, log, ptx) =
        binary("kernel k(Out: GlobalMemory<I32>) {\n    store(Out, 0, 7);\n}");
    assert!(!ok, "store(Out, 0, 7) must fail the build:\n{}", log);
    assert!(ptx.is_none(), "a refused build wrote a .ptx:\n{}", log);
    assert!(log.contains("`store(...)`") && log.contains("was given 3"), "{}", log);
}

/// On the card, the spelling the refusal recommends writes exactly the
/// element it names and nothing beside it, and the ten-argument 3-D store
/// writes its tenth argument - the operand a four-argument call used to read
/// from `%f0`.
#[test]
fn the_accepted_spellings_write_what_they_name() {
    use y::cuda_runtime::CudaContext;
    let (ok, log, ptx) = binary(
        "kernel k(Out: GlobalMemory<I32>, F: GlobalMemory<F32>) {\n    \
         store(Out[1], 7);\n    \
         block_ptr3d_store(F, 0, 0, 2, 1, 1, 1, 1, 4, 2.5);\n}",
    );
    assert!(ok, "the accepted spellings must compile:\n{}", log);
    let ptx = ptx.expect("no .ptx written");
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - the accepted spellings were not run on the device.");
        return;
    };
    let module = ctx.load_ptx(&ptx, "k").expect("PTX failed to load");
    let out = ctx.alloc(12).unwrap();
    let f = ctx.alloc(16).unwrap();
    ctx.memset_u8(&out, 0xAB).unwrap();
    ctx.memset_u8(&f, 0xAB).unwrap();
    ctx.launch(&module, (1, 1, 1), (1, 1, 1), 0, &[out.device_ptr(), f.device_ptr()])
        .expect("launch failed");
    ctx.synchronize().expect("kernel did not complete");
    let mut a = vec![0u8; 12];
    let mut b = vec![0u8; 16];
    ctx.memcpy_dtoh_at(&mut a, &out, 0).unwrap();
    ctx.memcpy_dtoh_at(&mut b, &f, 0).unwrap();
    let word = |v: &[u8], i: usize| u32::from_le_bytes(v[i * 4..i * 4 + 4].try_into().unwrap());
    let poison = u32::from_le_bytes([0xAB; 4]);
    assert_eq!([word(&a, 0), word(&a, 1), word(&a, 2)], [poison, 7, poison], "store(Out[1], 7)");
    assert_eq!(
        [word(&b, 0), word(&b, 1), word(&b, 2), word(&b, 3)],
        [poison, poison, 2.5f32.to_bits(), poison],
        "block_ptr3d_store's tenth argument lands at d2 = 2"
    );
}

/// The same built-in in the host backend. `--emit-cpu` lowered `store` to
/// `{place}.store_aligned_ptr({value} as *mut f32)` - a method of the deleted
/// `avx_wrapper` module, with the value cast to a pointer - and indexed its
/// second argument unconditionally, so `store(Out)` PANICKED the compiler
/// (exit 101) and `store(Out, 0, 7)` dropped the 7. It is refused by name now,
/// like the other GPU intrinsics that backend has no host equivalent for.
#[test]
fn the_host_backend_refuses_store_instead_of_crashing() {
    for body in ["store(Out[1], 7);", "store(Out, 0, 7);", "store(Out);"] {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "y_builtin_arity_cpu_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("k.ysu");
        std::fs::write(
            &file,
            format!("kernel k(Out: GlobalMemory<I32>) {{\n    {}\n}}\nfn main() {{}}\n", body),
        )
        .unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_Y"))
            .arg(&file)
            .arg("--emit-cpu")
            .current_dir(repo)
            .output()
            .expect("run Y");
        let log = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(out.status.code(), Some(1), "`{}` must be refused, not crash:\n{}", body, log);
        assert!(
            log.contains("`store`") && log.contains("GPU intrinsic"),
            "`{}` must be refused by name:\n{}",
            body,
            log
        );
        assert!(!log.contains("store_aligned_ptr"), "the dead lowering is still emitted:\n{}", log);
    }
}
