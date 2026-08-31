//! `SmemLayout` is documented as an API and is lowered by no backend.
//!
//! `docs/y_language_documentation.md` §21 is a full reference for
//! `SmemLayout<T, rows, cols, swizzle>` — a parameter table, a swizzle
//! formula, and a table of "common swizzle values" per tile width. Nothing in
//! `tests/*.ysu` declares one, so the whole surface was reachable from the
//! surface syntax and exercised by nothing: the profile the Hopper intrinsics
//! had before they were deleted.
//!
//! Three spellings, three different failures, and only one of them was
//! fail-closed:
//!
//!   - the inline form the document uses,
//!     `SharedMemory::alloc<SmemLayout<F32, rows=8, cols=32, swizzle=0>>()`,
//!     is a **syntax error**;
//!   - the alias form is refused by `--emit-ptx` (and by the host backends),
//!     which is correct;
//!   - **a kernel PARAMETER typed as a tile compiled clean**, printed
//!     "Compilation Successful!", exited 0, and emitted
//!     `add.u64 %rd3, %r0, %rd2` — a `.b32` register added to a `.b64` one —
//!     which `ptxas` rejects with "Arguments mismatch for instruction 'add'".
//!     The indexed read produced no value either, so the store wrote a literal
//!     `0`, and the address shift was 4 bytes for a 2-byte element type.
//!
//! The cause is the design-rule table's shape at the ABI boundary: kernel
//! parameter types were matched with `_ => ".param .b32"`, so a type the
//! backend cannot lower took a 32-bit slot and the body used it as an address.
//! `PtxEmitter::param_slot` refuses instead.
//!
//! `--emit-cpu` had the same class in call position, and worse, because that
//! backend prints Rust for a human to **paste** — Y never compiles the output,
//! so nothing downstream gets a chance to object. Four substitutions that
//! computed something other than what was asked: an F16 tile allocated as a
//! fixed 8192-element f32 buffer, `cp_async`'s byte count discarded (and in
//! the wrong unit), `ldmatrix` as an 8-wide f32 load, `mma_sync` as a vector
//! FMA. All four refused by name now.
//!
//! **The control matters as much as the census.** "Refuse every shared-memory
//! construct" would satisfy every negative assertion here and delete a working,
//! device-tested surface — so `the_working_shared_memory_surface_still_compiles`
//! pins that `shared_alloc_u32` / `shared_load_v4` / `barrier_sync` still do.
//!
//! **One mutation survives and is a confirmation rather than a hole.**
//! `param_slot` is called from two sites - the main entry and the split
//! paged-decode shape's second `.visible .entry` - because the standing rule
//! is to enumerate the sites, not the match arms. Reverting the second alone
//! is caught by nothing, and cannot be: the main entry is emitted first over
//! the same parameter list, so it refuses and the compile aborts before the
//! reduce entry exists. It is defence, and the code says so.
//!
//! Run with:  cargo test --release --test shared_memory_surface

use std::path::PathBuf;
use std::process::Command;

fn dir(tag: &str) -> PathBuf {
    // Per-test directory: these run in one binary and share a pid. The tag is
    // in the signature rather than in a comment because this race has fired
    // five times in this repository.
    let d = std::env::temp_dir().join(format!("y_smem_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("temp dir");
    d
}

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Compile `src` with `flag`, in a private directory so nothing writes an
/// artifact next to a committed one.
fn compile(tag: &str, src: &str, flag: &str) -> (bool, String, PathBuf) {
    let d = dir(tag);
    // The emitters read `.ysu_hw_profile` from the working directory; without
    // it they probe the GPU, which is slow and machine-dependent.
    let prof = repo().join(".ysu_hw_profile");
    if prof.exists() {
        let _ = std::fs::copy(&prof, d.join(".ysu_hw_profile"));
    }
    let f = d.join("probe.ysu");
    std::fs::write(&f, src).expect("write probe");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg("probe.ysu")
        .arg(flag)
        .current_dir(&d)
        .output()
        .expect("run Y");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text, d)
}

const ALIAS_FORM: &str = r#"
kernel k(C: GlobalMemory<F16>) {
    type ATile = SmemLayout<F16, rows=16, cols=64, swizzle=330>;
    let t = SharedMemory::alloc<ATile>();
    store(C, 0, 1.0);
}

fn main() {}
"#;

/// The spelling that compiled clean and produced PTX `ptxas` rejects.
const PARAM_FORM: &str = r#"
kernel k(T: SmemLayout<F16, rows=16, cols=64>, C: GlobalMemory<F16>) {
    let v: F16 = T[3];
    store(C, 0, v);
}

fn main() {}
"#;

const BACKENDS: &[&str] = &["--emit-ptx", "--emit-llvm", "--emit-native", "--emit-cpu"];

#[test]
fn no_backend_accepts_a_shared_memory_tile() {
    let mut checked = 0;
    for (i, flag) in BACKENDS.iter().enumerate() {
        for (j, (name, src)) in [("alias", ALIAS_FORM), ("param", PARAM_FORM)]
            .iter()
            .enumerate()
        {
            let (ok, text, _d) = compile(&format!("census{i}{j}"), src, flag);
            assert!(
                !ok,
                "{flag} accepted the {name} form of `SmemLayout`. Every backend \
                 either cannot lower it or lowers it to something that computes \
                 a different program; accepting it exits 0 and hands the user an \
                 artifact.\n{text}"
            );
            checked += 1;
        }
    }
    // A census in which nothing ran asserts nothing.
    assert_eq!(checked, BACKENDS.len() * 2, "the census did not run");
}

#[test]
fn the_tile_parameter_is_refused_by_name() {
    let (ok, text, _d) = compile("param", PARAM_FORM, "--emit-ptx");
    assert!(!ok, "the tile parameter is accepted again:\n{text}");
    assert!(
        text.contains("SmemLayout<...>") && text.contains("`.param` slot"),
        "the refusal must name the parameter's type and the reason, not fail \
         somewhere incidental - this program used to reach `ptxas`, and a \
         message about something else would leave the next reader guessing.\n{text}"
    );
    assert!(
        text.contains("shared_alloc_u32"),
        "the refusal must name the surface that works; a named gap costs five \
         minutes and a bare refusal costs an afternoon.\n{text}"
    );
}

#[test]
fn the_host_backend_refuses_the_gpu_intrinsics_it_used_to_fake() {
    // `--emit-cpu` prints Rust for a human to paste, so a substitution that
    // computes something else reaches their source with no compiler in between.
    for (i, (call, needle)) in [
        ("let tok = cp_async(a, b, 4); pipe.wait(tok);", "cp_async"),
        ("let x = ldmatrix(a);", "ldmatrix"),
        ("let y = mma_sync(a, b, a);", "mma_sync"),
    ]
    .iter()
    .enumerate()
    {
        let src = format!(
            "kernel k(A: GlobalMemory<F32>, B: GlobalMemory<F32>) {{\n    \
             let a = A;\n    let b = B;\n    {call}\n}}\n\nfn main() {{}}\n"
        );
        let (ok, text, _d) = compile(&format!("host{i}"), &src, "--emit-cpu");
        assert!(!ok, "--emit-cpu accepted `{needle}`:\n{text}");
        assert!(
            text.contains(needle),
            "the refusal must name `{needle}`.\n{text}"
        );
    }
}

/// The control. Without it, "refuse everything named shared" passes every
/// assertion above and deletes a surface that is tested on the device.
#[test]
fn the_working_shared_memory_surface_still_compiles() {
    let src = r#"
kernel k(Out: GlobalMemory<U32>) {
    let smem: U64 = shared_alloc_u32(64);
    let t: I32 = thread_idx_x();
    shared_store_v4(smem, t, t, t, t, t);
    barrier_sync();
    let v: U32x4 = shared_load_v4(smem, 0);
    block_ptr2d_store(Out, t, 0, 1, 1, 1, v.x);
}

fn main() {
}
"#;
    let (ok, text, d) = compile("control", src, "--emit-ptx");
    assert!(
        ok,
        "the real shared-memory surface stopped compiling. Refusing `SmemLayout` \
         must not take `shared_alloc_u32` with it - that path assembles AND runs \
         a cross-thread exchange on the device in tests/ptx_shared_memory.rs.\n{text}"
    );
    assert!(
        d.join("probe.ptx").exists(),
        "the backend reported success and wrote no module.\n{text}"
    );
}
