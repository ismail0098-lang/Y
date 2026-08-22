//! The shared-memory surface of the PTX backend: `shared_alloc_u32`,
//! `shared_load_v4`, `shared_store_v4`, `barrier_sync`.
//!
//! It exists because the NTT is DRAM-bound at 91% of peak, so the only lever
//! left on it is running fewer passes, and fewer passes means keeping
//! intermediates in shared memory across butterfly stages. Before this there
//! was no way to express that: `SharedMemory::alloc` emitted a hardcoded
//! 8 KB `.b8` blob with no way to index it, and **`barrier_sync()` emitted
//! nothing at all** while the barrier-hoisting pass in `emit_block` happily
//! recognised it and moved work across it.
//!
//! Three things are checked, and the second is the one that matters:
//!
//!  1. The PTX assembles (`ptxas`), so the declarations and addressing are
//!     structurally legal.
//!  2. It produces the right answer ON THE DEVICE, reading a slot written by
//!     a different thread. A kernel where every thread reads back its own
//!     write is satisfied by no shared memory, no store and no barrier.
//!  3. An illegal allocation is refused with a reason rather than deferred to
//!     a ptxas message about the whole module.
//!
//! Run with:  cargo test --release --test ptx_shared_memory

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("Y")
}

fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Compile a `.ysu` from `tests/` and return its PTX.
///
/// Compiled once per process behind a lock: two tests asking for the same
/// entry race on the `.ptx` output path, one reading it while the other is
/// still writing. That is the failure `zk_gpu_ntt.rs::ptx_for` documents, and
/// it showed up here immediately as "no barrier in the probe" against a file
/// that plainly had one.
fn compile(entry: &str) -> String {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<std::collections::HashMap<String, String>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = guard.get(entry) {
        return p.clone();
    }
    let out = Command::new(bin())
        .arg(repo().join(format!("tests/{}.ysu", entry)))
        .arg("--emit-ptx")
        .current_dir(repo())
        .output()
        .expect("failed to run the Y binary");
    assert!(
        out.status.success(),
        "{}.ysu did not compile:\n{}{}",
        entry,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read_to_string(repo().join(format!("tests/{}.ptx", entry)))
        .expect("no .ptx written");
    guard.insert(entry.to_string(), ptx.clone());
    ptx
}

/// Compile arbitrary source through the real binary, returning
/// `Ok(stdout)` or `Err(stdout+stderr)`.
fn compile_source(tag: &str, src: &str) -> Result<String, String> {
    let dir = std::env::temp_dir().join(format!("y_smem_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("k.ysu");
    std::fs::write(&path, src).expect("write Y source");
    let out = Command::new(bin())
        .arg(&path)
        .arg("--emit-ptx")
        .current_dir(repo())
        .output()
        .expect("failed to run the Y binary");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
    if out.status.success() {
        Ok(text)
    } else {
        Err(text)
    }
}

fn ptxas(ptx: &str, tag: &str) {
    let dir = std::env::temp_dir().join(format!("y_smem_asm_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("k.ptx");
    std::fs::write(&src, ptx).unwrap();
    let out = match Command::new("ptxas")
        .arg("-arch=sm_89")
        .arg(&src)
        .arg("-o")
        .arg(dir.join("k.cubin"))
        .output()
    {
        Ok(o) => o,
        Err(_) => {
            eprintln!("SKIP: no ptxas on PATH — {} was not assembled.", tag);
            return;
        }
    };
    assert!(
        out.status.success(),
        "{} does not assemble:\n{}",
        tag,
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_shared_memory_probe_assembles() {
    let ptx = compile("smem_roundtrip");
    assert!(
        ptx.contains(".shared .align 16 .b32 __y_smem_0[2048];"),
        "no `.shared` array was declared:\n{}",
        &ptx[..ptx.len().min(2000)]
    );
    assert!(ptx.contains("st.shared.v4.u32"), "no shared store emitted");
    assert!(ptx.contains("ld.shared.v4.u32"), "no shared load emitted");
    assert!(ptx.contains("bar.sync 0;"), "no barrier emitted");
    ptxas(&ptx, "smem_roundtrip");
}

/// The regression this file was written for.
///
/// `emit_block`'s barrier-hoisting pass moves arithmetic from after a barrier
/// to before it. It used to do so with no dependency check at all AND without
/// stopping at the first statement it could not hoist, so it reached past the
/// loads defining its operands. In the fused NTT that produced PTX referring
/// to `qb0`, `wb3` and friends as undefined symbols.
///
/// Assembling is the real gate — an unresolved operand is exactly what ptxas
/// rejects — but assert the ordering directly too, because a future version
/// of the pass could hoist something that happens to resolve to the WRONG
/// register rather than to nothing.
#[test]
fn hoisting_cannot_cross_a_real_dependency() {
    let ptx = compile("smem_roundtrip");
    let bar = ptx.find("bar.sync 0;").expect("no barrier in the probe");
    let ld = ptx.find("ld.shared.v4.u32").expect("no shared load in the probe");
    assert!(
        bar < ld,
        "a shared load was emitted before the barrier that orders it"
    );
    // `acc = v0 + 7;` reads a value that only exists after the barrier. The
    // hoisting pass must have left it where it is.
    // The literal is materialised into a register at the point the statement
    // is emitted (`mov.u32 %rN, 7;`), so that `mov` marks where the statement
    // ended up.
    let add7 = ptx
        .lines()
        .position(|l| l.trim_start().starts_with("mov.u32") && l.trim_end().ends_with(", 7;"))
        .expect("the +7 that pins the hoist is gone — the probe no longer tests anything");
    let bar_line = ptx
        .lines()
        .position(|l| l.contains("bar.sync 0;"))
        .unwrap();
    assert!(
        add7 > bar_line,
        "an ALU op consuming a post-barrier value was hoisted above the barrier"
    );
}

/// The one that cannot be faked: run it.
///
/// Thread `t` writes `base*4 .. base*4+3` into its own slot and then reads the
/// slot of thread `255 - t`. The answer therefore names a value this thread
/// never held, so a kernel with the store removed, or with shared memory not
/// existing at all, cannot produce it.
///
/// **It does NOT prove the barrier is emitted, and that was checked rather
/// than assumed**: with `bar.sync` mutated away this test still passes, every
/// run. The race is real but does not fire, because the eight warps happen to
/// issue their stores before any of them reaches its load. A missing barrier
/// is caught by `the_shared_memory_probe_assembles` and
/// `hoisting_cannot_cross_a_real_dependency` — which do fail under that
/// mutation — and not here. Do not delete those in favour of "we run it".
#[test]
fn the_probe_computes_the_right_answer_on_the_device() {
    use y::cuda_runtime::CudaContext;

    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — smem_roundtrip.ysu was not executed.");
        return;
    };
    let ptx = compile("smem_roundtrip");
    let module = ctx
        .load_ptx(&ptx, "smem_roundtrip")
        .expect("smem_roundtrip did not load");

    const BLOCKS: u32 = 4;
    const THREADS: u32 = 256;
    let n = (BLOCKS * THREADS) as usize;

    let d_out = ctx.alloc(n * 4 * 4).unwrap();
    ctx.memcpy_htod_at(&d_out, 0, &vec![0u8; n * 4 * 4]).unwrap();
    let args = vec![d_out.device_ptr(), n as u64];
    ctx.launch(&module, (BLOCKS, 1, 1), (THREADS, 1, 1), 0, &args)
        .expect("launch failed");
    ctx.synchronize().unwrap();

    let mut raw = vec![0u8; n * 4 * 4];
    ctx.memcpy_dtoh_at(&mut raw, &d_out, 0).unwrap();
    let got: Vec<u32> = raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    for b in 0..BLOCKS as usize {
        for t in 0..THREADS as usize {
            let base = b * 256 + t;
            let peer = b * 256 + (255 - t);
            let want = [
                (peer * 4) as u32 + 7,
                (peer * 4 + 1) as u32,
                (peer * 4 + 2) as u32,
                (peer * 4 + 3) as u32,
            ];
            for (k, w) in want.iter().enumerate() {
                assert_eq!(
                    got[base * 4 + k],
                    *w,
                    "block {} thread {} lane {}: shared-memory exchange is wrong",
                    b,
                    t,
                    k
                );
            }
        }
    }
}

/// A `.shared` array's size is fixed when the module is assembled, so a
/// runtime count cannot be honoured. Refusing names the cause; accepting and
/// guessing a size would be the failure mode CLAUDE.md's design rule is about.
#[test]
fn a_runtime_allocation_size_is_refused() {
    let src = r#"
kernel bad(Out: GlobalMemory<U32>, N: I32) {
    let smem: U64 = shared_alloc_u32(N);
    let t: I32 = thread_idx_x();
    shared_store_v4(smem, t, t, t, t, t);
}

fn main() {
}
"#;
    let err = compile_source("runtime_size", src)
        .expect_err("a runtime shared-memory size was accepted");
    assert!(
        err.contains("shared_alloc_u32"),
        "the refusal does not name the intrinsic:\n{}",
        err
    );
}

#[test]
fn an_over_large_allocation_is_refused() {
    let src = r#"
kernel bad(Out: GlobalMemory<U32>, N: I32) {
    let smem: U64 = shared_alloc_u32(16384);
    let t: I32 = thread_idx_x();
    shared_store_v4(smem, t, t, t, t, t);
}

fn main() {
}
"#;
    let err = compile_source("too_big", src).expect_err("a 64 KB static allocation was accepted");
    assert!(
        err.contains("48 KB") || err.contains("shared_alloc_u32"),
        "the refusal does not explain the limit:\n{}",
        err
    );
}

/// The control for the two refusals above: the same shape at a legal size must
/// still compile. Without this, "shared memory is refused unconditionally"
/// would pass both.
#[test]
fn a_legal_allocation_is_accepted() {
    let src = r#"
kernel ok(Out: GlobalMemory<U32>, N: I32) {
    let smem: U64 = shared_alloc_u32(1024);
    let t: I32 = thread_idx_x();
    shared_store_v4(smem, t, t, t, t, t);
}

fn main() {
}
"#;
    compile_source("legal", src).expect("a legal shared-memory allocation was refused");
}

/// Two kernels in one module must not both get `__y_smem_0`. The symbol is
/// module-scope in PTX, so a collision is a redefinition ptxas rejects — and
/// it only ever appears in a module with two shared-memory kernels in it.
#[test]
fn two_kernels_get_distinct_shared_symbols() {
    let src = r#"
kernel first(Out: GlobalMemory<U32>, N: I32) {
    let smem: U64 = shared_alloc_u32(1024);
    let t: I32 = thread_idx_x();
    shared_store_v4(smem, t, t, t, t, t);
}

kernel second(Out: GlobalMemory<U32>, N: I32) {
    let smem: U64 = shared_alloc_u32(512);
    let t: I32 = thread_idx_x();
    shared_store_v4(smem, t, t, t, t, t);
}

fn main() {
}
"#;
    compile_source("two_kernels", src).expect("two shared-memory kernels did not compile");
    // The emitter writes `<stem>.ptx` next to the source; compile_source
    // deletes the directory, so re-read through a persistent path instead.
    let dir = std::env::temp_dir().join(format!("y_smem_two_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("k.ysu");
    std::fs::write(&path, src).unwrap();
    let out = Command::new(bin())
        .arg(&path)
        .arg("--emit-ptx")
        .current_dir(repo())
        .output()
        .unwrap();
    assert!(out.status.success());
    let ptx = std::fs::read_to_string(dir.join("k.ptx")).unwrap();
    assert!(
        ptx.contains("__y_smem_0[1024]") && ptx.contains("__y_smem_1[512]"),
        "the two kernels did not get distinct `.shared` symbols:\n{}",
        ptx.lines()
            .filter(|l| l.contains(".shared"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    ptxas(&ptx, "two_kernels");
    let _ = std::fs::remove_dir_all(&dir);
}
