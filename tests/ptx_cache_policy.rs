//! `@cache_policy(P)` on a PTX load emits the instruction the PTX ISA gives
//! for what `P` means, or is refused. It did neither:
//!
//! | policy | language reference (9.2) | emitted before | what that is |
//! | --- | --- | --- | --- |
//! | `L2_EVICT_FIRST` | evict as soon as possible | `ld.global.L2::evict_first.f32` | rejected by `ptxas` at every arch: `ld` takes that qualifier only for `.v8.b32`/`.v4.b64` (sm_100) |
//! | `L2_PERSIST` | stay resident in L2 | `ld.global.lu` | on a global address "a load cached streaming operation (ld.cs)" - evict FIRST |
//! | `L2_EVICT_LAST` | prevent early eviction | `ld.global.ca` | the directive dropped |
//! | `L2_STREAM` | read once | `ld.global.ca` | the directive dropped |
//!
//! Now: `L2_STREAM` is `ld.global.cs`; `L2_EVICT_FIRST`, `L2_EVICT_LAST` and
//! `L2_PERSIST` are an L2 eviction priority, `createpolicy.fractional.L2::<p>`
//! plus `ld.global.L2::cache_hint` (sm_80 and PTX 7.4, which the module
//! declares when it needs them; refused below sm_80). A policy with nothing to
//! apply to - on a store, on a `let` whose load ignores it - is refused
//! rather than parsed and dropped.
//!
//! The quoted ISA text is from the PTX ISA's cache-operator table and the `ld`
//! and `createpolicy` pages; the rejection is measured with `ptxas`.

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Emitted {
    ok: bool,
    log: String,
    ptx: Option<String>,
}

/// Compiles `src` (`fn main() {}` appended) with `--emit-ptx` for `sm`
/// (e.g. "8.9"), pinned by a `.ysu_hw_profile` in its own temp dir - the
/// device `ptx_portability` uses, so the build machine's card is not the arch.
fn emit_for(sm: &str, tag: &str, src: &str) -> Emitted {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "y_cache_policy_{}_{}_{}",
        std::process::id(),
        tag,
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join(".ysu_hw_profile"),
        format!("SM_VERSION={}\nGPU_NAME=TestCard\nSM_COUNT=66\n", sm),
    )
    .unwrap();
    std::fs::write(dir.join("k.ysu"), format!("{}\nfn main() {{}}\n", src)).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg("k.ysu")
        .arg("--emit-ptx")
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read_to_string(dir.join("k.ptx")).ok();
    let _ = std::fs::remove_dir_all(&dir);
    Emitted { ok: out.status.success(), log, ptx }
}

fn compiled_for(sm: &str, tag: &str, src: &str) -> String {
    let e = emit_for(sm, tag, src);
    assert!(e.ok, "{} must compile for sm {}:\n{}", tag, sm, e.log);
    e.ptx.unwrap_or_else(|| panic!("{} compiled but wrote no .ptx:\n{}", tag, e.log))
}

fn refused(tag: &str, src: &str, needles: &[&str]) {
    let e = emit_for("8.9", tag, src);
    assert!(!e.ok, "{} must be refused:\n{}", tag, e.log);
    assert!(e.ptx.is_none(), "{} was refused but still wrote a .ptx", tag);
    for n in needles {
        assert!(e.log.contains(n), "{}: the refusal must say `{}`:\n{}", tag, n, e.log);
    }
}

fn ptxas_present() -> bool {
    Command::new("ptxas").arg("--version").output().is_ok()
}

/// Runs `ptxas -arch=<the module's own .target>` over `ptx`.
fn assembles(tag: &str, ptx: &str) -> Result<(), String> {
    let target = ptx
        .lines()
        .find_map(|l| l.trim().strip_prefix(".target "))
        .expect("module declares no .target")
        .trim()
        .to_string();
    let f = std::env::temp_dir().join(format!("y_cache_policy_{}_{}.ptx", std::process::id(), tag));
    std::fs::write(&f, ptx).unwrap();
    let out = Command::new("ptxas")
        .arg(format!("-arch={}", target))
        .arg(&f)
        .arg("-o")
        .arg("/dev/null")
        .output()
        .expect("run ptxas");
    let _ = std::fs::remove_file(&f);
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).to_string())
    }
}

fn loader(policy: Option<&str>, read: &str) -> String {
    format!(
        "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>) {{\n{}    let v: F32 = {};\n    Out[0] = v;\n}}",
        policy.map(|p| format!("    @cache_policy({})\n", p)).unwrap_or_default(),
        read
    )
}

/// (policy, the instruction(s) the load must be) - one row per meaning.
const FORMS: &[(Option<&str>, &[&str])] = &[
    (None, &["ld.global.ca.f32"]),
    (Some("L2_STREAM"), &["ld.global.cs.f32"]),
    (
        Some("L2_EVICT_FIRST"),
        &["createpolicy.fractional.L2::evict_first.b64", "ld.global.L2::cache_hint.f32"],
    ),
    (
        Some("L2_EVICT_LAST"),
        &["createpolicy.fractional.L2::evict_last.b64", "ld.global.L2::cache_hint.f32"],
    ),
    (
        Some("L2_PERSIST"),
        &["createpolicy.fractional.L2::evict_last.b64", "ld.global.L2::cache_hint.f32"],
    ),
];

/// Every policy, through both loads that honour one, is the ISA's form for
/// its meaning - and no longer `.lu` or the `.L2::evict_first` qualifier.
#[test]
fn every_policy_emits_the_isa_form_for_its_meaning() {
    for (policy, want) in FORMS {
        for read in ["A[1]", "GlobalMemory::load(A[1])"] {
            let ptx = compiled_for("8.9", "form", &loader(*policy, read));
            for w in *want {
                assert!(ptx.contains(w), "{:?} via `{}` must emit `{}`:\n{}", policy, read, w, ptx);
            }
            assert!(!ptx.contains(".lu."), "{:?}: `.lu` is evict-first on a global address:\n{}", policy, ptx);
            assert!(
                !ptx.contains("ld.global.L2::evict"),
                "{:?}: `ld` takes `.L2::evict_*` only on 256-bit vectors:\n{}",
                policy,
                ptx
            );
        }
    }
}

/// The v4 load honours the policy the same way.
#[test]
fn the_v4_load_takes_the_same_policy() {
    let ptx = compiled_for(
        "8.9",
        "v4",
        "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>) {\n    \
         @cache_policy(L2_EVICT_FIRST)\n    \
         let v: F32 = ld_global_v4_f32(A);\n    \
         Out[0] = v;\n}",
    );
    assert!(ptx.contains("createpolicy.fractional.L2::evict_first.b64"), "{}", ptx);
    assert!(ptx.contains("ld.global.L2::cache_hint.v4.f32"), "{}", ptx);
}

/// Every form assembles at every arch the backend targets from sm_80 up - the
/// reported bug was a module `ptxas` rejects after a clean compile. One read
/// spelling is enough here: both go through one load helper, and the test
/// above pins that they emit the same instructions.
#[test]
fn every_policy_assembles_at_every_supported_arch() {
    if !ptxas_present() {
        eprintln!("SKIP: ptxas not on PATH - the cache-policy forms were not assembled.");
        return;
    }
    let mut failures = Vec::new();
    for sm in ["8.0", "8.6", "8.9", "9.0", "12.0"] {
        for (policy, _) in FORMS {
            let ptx = compiled_for(sm, "asm", &loader(*policy, "A[1]"));
            if let Err(e) = assembles("asm", &ptx) {
                failures.push(format!("sm {} {:?}: {}", sm, policy, e.trim()));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `createpolicy` needs PTX 7.4, and sm_80's floor is 7.0: a module using it
/// declares 7.4, and one that does not keeps the floor.
#[test]
fn the_module_declares_the_version_createpolicy_needs() {
    let version = |ptx: &str| {
        ptx.lines()
            .find_map(|l| l.trim().strip_prefix(".version ").map(|v| v.trim().to_string()))
            .unwrap()
    };
    assert_eq!(version(&compiled_for("8.0", "v_plain", &loader(None, "A[1]"))), "7.0");
    assert_eq!(version(&compiled_for("8.0", "v_stream", &loader(Some("L2_STREAM"), "A[1]"))), "7.0");
    assert_eq!(version(&compiled_for("8.0", "v_first", &loader(Some("L2_EVICT_FIRST"), "A[1]"))), "7.4");
}

/// Below sm_80 there is no `createpolicy`: an L2 priority is refused by name,
/// and `L2_STREAM` (`.cs`, every target has it) still compiles.
#[test]
fn an_l2_priority_below_sm80_is_refused() {
    for p in ["L2_EVICT_FIRST", "L2_EVICT_LAST", "L2_PERSIST"] {
        let e = emit_for("7.5", "old", &loader(Some(p), "A[1]"));
        assert!(!e.ok && e.ptx.is_none(), "{} at sm_75 must be refused:\n{}", p, e.log);
        assert!(e.log.contains("createpolicy") && e.log.contains("sm_75"), "{}:\n{}", p, e.log);
    }
    let ptx = compiled_for("7.5", "old_stream", &loader(Some("L2_STREAM"), "A[1]"));
    assert!(ptx.contains("ld.global.cs.f32"), "{}", ptx);
}

/// A policy nothing would honour is refused: on a store (the language
/// reference's own `@cache_policy(L2_STREAM) C[i] = a_val + b_val;`), on a
/// `let` whose load ignores it, and on a `let` that loads nothing.
#[test]
fn a_policy_nothing_would_honour_is_refused() {
    refused(
        "on_store",
        "kernel k(A: GlobalMemory<F32>, C: GlobalMemory<F32>) {\n    \
         @cache_policy(L2_STREAM)\n    \
         C[0] = 1.0;\n}",
        &["applies to a `let`", "silently ignored"],
    );
    refused(
        "ignoring_load",
        "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>) {\n    \
         @cache_policy(L2_PERSIST)\n    \
         let v: F32 = block_ptr2d_load(A, 0, 0, 1, 1, 1);\n    \
         Out[0] = v;\n}",
        &["block_ptr2d_load", "does not take a cache policy"],
    );
    refused(
        "no_load",
        "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>) {\n    \
         @cache_policy(L2_STREAM)\n    \
         let v: F32 = 2.0;\n    \
         Out[0] = v;\n}",
        &["loads nothing"],
    );
}

/// `reuse_count=N` was parsed and dropped by every backend, and the manual said
/// so. A directive that changes nothing is refused by name, whichever policy
/// carries it; the same policy without the count still compiles.
#[test]
fn a_reuse_count_nothing_lowers_is_refused() {
    for (tag, policy) in [("reuse_persist", "L2_PERSIST, reuse_count=8"), ("reuse_stream", "L2_STREAM, reuse_count=2")] {
        refused(tag, &loader(Some(policy), "A[1]"), &["reuse_count", "silently ignored"]);
    }
    let ptx = compiled_for("8.9", "reuse_control", &loader(Some("L2_PERSIST"), "A[1]"));
    assert!(ptx.contains("createpolicy.fractional.L2::evict_last.b64"), "{}", ptx);
}

/// A cache hint changes where a line lives, never what a load returns: every
/// policy reads back element 1 exactly.
#[test]
fn a_policy_does_not_change_the_value_loaded() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - the cache-policy loads were not run.");
        return;
    };
    for (policy, _) in FORMS {
        let ptx = compiled_for("8.9", "dev", &loader(*policy, "A[1]"));
        let module = ctx.load_ptx(&ptx, "k").expect("PTX failed to load");
        let a = ctx.alloc(12).unwrap();
        let o = ctx.alloc(8).unwrap();
        let init: Vec<u8> = [1.5f32, 2.25, -3.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        ctx.memcpy_htod_at(&a, 0, &init).unwrap();
        ctx.memcpy_htod_at(&o, 0, &[0xABu8; 8]).unwrap();
        ctx.launch(&module, (1, 1, 1), (1, 1, 1), 0, &[a.device_ptr(), o.device_ptr()])
            .expect("launch failed");
        ctx.synchronize().expect("kernel did not complete");
        let mut out = vec![0u8; 8];
        ctx.memcpy_dtoh_at(&mut out, &o, 0).unwrap();
        assert_eq!(&out[..4], &2.25f32.to_le_bytes(), "{:?}", policy);
        assert_eq!(&out[4..], &[0xAB; 4], "{:?}", policy);
    }
}
