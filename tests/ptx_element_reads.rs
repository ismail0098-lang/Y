//! Reading `A[i]` in a PTX kernel loads element `i`.
//!
//! It used to evaluate to the element's ADDRESS in every position, so
//!
//! ```text
//! kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>) {
//!     let v: F32 = A[1];
//!     Out[0] = v * 2.0;
//! }
//! ```
//!
//! emitted `cvt.rn.f32.u64 %f0, %rd4` - the address of `A[1]`, converted to a
//! float - under a clean compile, exit 0, and a module `ptxas` accepts. The
//! address positions (`store`'s place, `GlobalMemory::load`'s argument, every
//! memory built-in's buffer, an indexed assignment target) now take the
//! address explicitly, and every other position loads.
//!
//! Two things moved with it and are pinned here too:
//!
//! * `Out[i] op= v` was refused while a read gave the address. It is a load,
//!   the operation and a store now, with the index evaluated once.
//! * The barrier-hoisting pass moved any `let x = a op b` whose operands were
//!   bound across `barrier_sync()`, calling it an "independent ALU
//!   instruction". An element read is a load and an element assignment a
//!   store, and a `GlobalMemory::load` inside arithmetic always was one. On
//!   the card, with the writing warps delayed, HEAD read the stale value in
//!   every launch.
//!
//! The device tests pre-fill every buffer with 0xAB and compare BYTES,
//! including the elements beside the ones a kernel names. The source-level
//! tests below them need no GPU.

use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Emitted {
    ok: bool,
    log: String,
    ptx: Option<String>,
}

/// Compiles `src` (a kernel named `k`; `fn main() {}` is appended) with
/// `--emit-ptx`, in its own temp dir: `--emit-ptx` writes next to its input,
/// and a per-test tag alone is not unique across tests in one file.
fn emit(tag: &str, src: &str) -> Emitted {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!(
        "y_element_reads_{}_{}_{}",
        std::process::id(),
        tag,
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
    Emitted { ok: out.status.success(), log, ptx }
}

fn compiled(tag: &str, src: &str) -> String {
    let e = emit(tag, src);
    assert!(e.ok, "{} must compile:\n{}", tag, e.log);
    e.ptx.unwrap_or_else(|| panic!("{} compiled but wrote no .ptx:\n{}", tag, e.log))
}

/// The instruction lines of a module: comments and blank lines dropped.
fn instructions(ptx: &str) -> Vec<&str> {
    ptx.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with("//"))
        .collect()
}

/// Launches `k` once over `block` threads in one block. `bufs` are the
/// initial contents of the kernel's buffer parameters, in order. Returns
/// every buffer's bytes afterwards, or `None` with no CUDA driver.
fn run(ptx: &str, bufs: &[Vec<u8>], block: u32) -> Option<Vec<Vec<u8>>> {
    use y::cuda_runtime::CudaContext;
    let ctx = CudaContext::new()?;
    let module = ctx.load_ptx(ptx, "k").expect("PTX failed to load");
    let mut dev = Vec::new();
    for init in bufs {
        let b = ctx.alloc(init.len()).unwrap();
        ctx.memcpy_htod_at(&b, 0, init).unwrap();
        dev.push(b);
    }
    let args: Vec<u64> = dev.iter().map(|b| b.device_ptr()).collect();
    ctx.launch(&module, (1, 1, 1), (block, 1, 1), 0, &args)
        .expect("launch failed");
    ctx.synchronize().expect("kernel did not complete");
    Some(
        dev.iter()
            .map(|b| {
                let mut v = vec![0u8; b.len_bytes()];
                ctx.memcpy_dtoh_at(&mut v, b, 0).unwrap();
                v
            })
            .collect(),
    )
}

fn skip(what: &str) {
    eprintln!("SKIP: no CUDA driver - {} was not checked on the device.", what);
}

const POISON: u8 = 0xAB;

fn poisoned(bytes: usize) -> Vec<u8> {
    vec![POISON; bytes]
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

const P32: u32 = 0xABAB_ABAB;

// ───────────────────────────── on the device ─────────────────────────────

/// The reported program. It stored `2 * float(&A[1])`; the element is 2.25.
#[test]
fn reading_an_element_loads_its_value() {
    let ptx = compiled(
        "reported",
        "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>) {\n    \
         let v: F32 = A[1];\n    \
         Out[0] = v * 2.0;\n}",
    );
    assert!(
        !ptx.contains("cvt.rn.f32.u64"),
        "an element read must not convert an address to a float:\n{}",
        ptx
    );
    let Some(out) = run(&ptx, &[f32_bytes(&[1.5, 2.25, -3.0]), poisoned(8)], 1) else {
        return skip("element read");
    };
    assert_eq!(words(&out[1]), [4.5f32.to_bits(), P32], "Out[0] = 2 * A[1]; Out[1] untouched");
    assert_eq!(out[0], f32_bytes(&[1.5, 2.25, -3.0]), "a read writes nothing");
}

/// `Out[0] = A[1]` for every element type: the element's bytes come back
/// exactly, and the element beside it is untouched. F16 is widened to f32 on
/// the load and rounded on the store, which is exact for any finite F16.
#[test]
fn every_element_width_reads_back_exactly() {
    // (type, element bytes, element 1's bytes)
    let cases: &[(&str, usize, &[u8])] = &[
        ("U8", 1, &[0xFB]),
        ("I8", 1, &[0x85]),
        ("U16", 2, &[0x34, 0xF2]),
        ("I16", 2, &[0x01, 0x80]),
        ("U32", 4, &[0x78, 0x56, 0x34, 0xF2]),
        ("I32", 4, &[0xFF, 0xFF, 0xFF, 0x80]),
        ("U64", 8, &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0xF8]),
        ("I64", 8, &[0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]),
        ("F32", 4, &[0x01, 0x00, 0xC0, 0x3F]),
        ("F64", 8, &[0x18, 0x2D, 0x44, 0x54, 0xFB, 0x21, 0x09, 0x40]),
        ("F16", 2, &[0x01, 0x3C]),
    ];
    for (ty, w, e1) in cases {
        let ptx = compiled(
            &format!("width_{}", ty),
            &format!(
                "kernel k(A: GlobalMemory<{0}>, Out: GlobalMemory<{0}>) {{\n    Out[0] = A[1];\n}}",
                ty
            ),
        );
        let mut a = vec![0x11u8; *w];
        a.extend_from_slice(e1);
        a.extend(std::iter::repeat(0x22u8).take(*w));
        let Some(out) = run(&ptx, &[a, poisoned(2 * w)], 1) else {
            return skip("element widths");
        };
        assert_eq!(&out[1][..*w], *e1, "{}: Out[0] must be A[1]'s bytes", ty);
        assert_eq!(&out[1][*w..], &vec![POISON; *w][..], "{}: Out[1] must be untouched", ty);
    }
}

/// A sub-word element read into an I32 is extended by the ELEMENT's
/// signedness: an I8 0x85 is -123 and a U8 0xFB is 251.
#[test]
fn a_subword_element_is_extended_by_its_own_signedness() {
    for (ty, byte, want) in [("I8", 0x85u8, -123i32), ("U8", 0xFB, 251)] {
        let ptx = compiled(
            &format!("ext_{}", ty),
            &format!(
                "kernel k(A: GlobalMemory<{}>, Out: GlobalMemory<I32>) {{\n    \
                 let v: I32 = A[1];\n    \
                 Out[0] = v;\n}}",
                ty
            ),
        );
        let Some(out) = run(&ptx, &[vec![0, byte, 0], poisoned(8)], 1) else {
            return skip("sub-word extension");
        };
        assert_eq!(words(&out[1]), [want as u32, P32], "{}", ty);
    }
}

/// Element reads as operands of arithmetic, of a negation and of a
/// condition, and as the index of another element.
#[test]
fn element_reads_compose_with_everything_else() {
    let ptx = compiled(
        "compose",
        "kernel k(A: GlobalMemory<F32>, I: GlobalMemory<U32>, Out: GlobalMemory<F32>) {\n    \
         Out[0] = A[1] * 2.0 + A[2];\n    \
         Out[1] = -A[3];\n    \
         if A[0] > 1.0 {\n        Out[2] = 1.0;\n    }\n    \
         if A[0] > 9.0 {\n        Out[3] = 1.0;\n    }\n    \
         Out[I[0]] = A[I[1]];\n}",
    );
    let a = f32_bytes(&[1.5, 2.25, -3.0, 7.0]);
    let Some(out) = run(&ptx, &[a, u32_bytes(&[5, 3]), poisoned(24)], 1) else {
        return skip("element reads in expressions");
    };
    assert_eq!(
        words(&out[2]),
        [
            (2.25f32 * 2.0 - 3.0).to_bits(),
            (-7.0f32).to_bits(),
            1.0f32.to_bits(),
            P32,
            P32,
            7.0f32.to_bits()
        ],
        "[A1*2+A2, -A3, A0>1, A0>9 untouched, untouched, Out[I0] = A[I1]]"
    );
}

/// `Out[i] op= v` loads, operates and stores - it was refused while a read
/// gave the address. The F16 case is a tie: 2050 + 1 = 2051 lies between the
/// F16 values 2050 (0x6801) and 2052 (0x6802), and nearest-even is 0x6802.
#[test]
fn compound_assignment_to_an_element_reads_modifies_and_writes() {
    let ptx = compiled(
        "compound",
        "kernel k(Out: GlobalMemory<F32>, C: GlobalMemory<U32>, H: GlobalMemory<F16>) {\n    \
         Out[1] += 2.5;\n    \
         Out[2] *= 2.0;\n    \
         C[0] -= 3;\n    \
         H[1] += 1.0;\n}",
    );
    let h: Vec<u8> = [0x3C00u16, 0x6801, 0x3C00].iter().flat_map(|x| x.to_le_bytes()).collect();
    let Some(out) = run(&ptx, &[f32_bytes(&[1.0, 2.0, 3.0, 4.0]), u32_bytes(&[5, 9]), h], 1) else {
        return skip("compound assignment");
    };
    assert_eq!(out[0], f32_bytes(&[1.0, 4.5, 6.0, 4.0]));
    assert_eq!(words(&out[1]), [2, 9]);
    let h: Vec<u16> = out[2].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    assert_eq!(h, [0x3C00, 0x6802, 0x3C00], "2050 + 1 rounds to nearest-even 2052");
}

/// The index of `Out[I[0]] += 1.0` is evaluated once: one load of `I`, one
/// load of `Out`, one store. Desugaring it as `Out[I[0]] = Out[I[0]] + 1.0`
/// evaluates it twice, which is only right while no index expression has an
/// effect.
#[test]
fn a_compound_assignment_evaluates_its_index_once() {
    let ptx = compiled(
        "once",
        "kernel k(Out: GlobalMemory<F32>, I: GlobalMemory<U32>) {\n    Out[I[0]] += 1.0;\n}",
    );
    let body = instructions(&ptx);
    let loads = body.iter().filter(|l| l.starts_with("ld.global")).count();
    let stores = body.iter().filter(|l| l.starts_with("st.global")).count();
    assert_eq!((loads, stores), (2, 1), "one load of I, one of Out, one store:\n{}", ptx);
    let Some(out) = run(&ptx, &[f32_bytes(&[10.0, 20.0, 30.0]), u32_bytes(&[2])], 1) else {
        return skip("index evaluated once");
    };
    assert_eq!(out[0], f32_bytes(&[10.0, 20.0, 31.0]));
}

/// A load after `barrier_sync()` must stay after it. The writers of the
/// elements a warp reads are delayed by a chain of dependent loads, so a read
/// moved above the barrier sees the poison: HEAD did, in every launch. The
/// control places the read before the barrier on purpose - if that sees no
/// stale value either, the race cannot fire on this card and the device half
/// says nothing (the ordering test below still holds).
#[test]
fn a_load_after_a_barrier_sees_what_other_warps_wrote() {
    fn kernel(read_before_barrier: bool) -> String {
        let chain = "        d = GlobalMemory::load(B[d]);\n".repeat(24);
        let read = "    let x: U32 = GlobalMemory::load(A[(t + 32) & 1023]) + z;\n";
        format!(
            "kernel k(A: GlobalMemory<U32>, Out: GlobalMemory<U32>, B: GlobalMemory<U32>) {{\n    \
             let t: U32 = thread_idx_x();\n    \
             let z: U32 = 0;\n    \
             let mut d: U32 = 0;\n    \
             if (t & 32) > z {{\n{}    }}\n    \
             A[t] = d + 1;\n{}    barrier_sync();\n{}    Out[t] = x;\n}}",
            chain,
            if read_before_barrier { read } else { "" },
            if read_before_barrier { "" } else { read },
        )
    }
    let stale = |ptx: &str| -> Option<usize> {
        let mut total = 0;
        for _ in 0..20 {
            let out = run(ptx, &[poisoned(4096), poisoned(4096), vec![0; 4096]], 1024)?;
            total += words(&out[1]).iter().filter(|&&w| w != 1).count();
        }
        Some(total)
    };
    let fixed = compiled("barrier_read", &kernel(false));
    let control = compiled("barrier_read_early", &kernel(true));
    let (Some(fixed_stale), Some(control_stale)) = (stale(&fixed), stale(&control)) else {
        return skip("cross-warp read after a barrier");
    };
    assert_eq!(fixed_stale, 0, "a read after the barrier saw another warp's element unwritten");
    if control_stale == 0 {
        eprintln!(
            "NOTE: the read-before-barrier control saw no stale value on this card, so \
             the device half of this test cannot see a hoisted read here."
        );
    }
}

/// A built-in handed an element address takes the element type of the
/// buffer it points into. `block_ptr2d_load` looked the type up by buffer
/// NAME only, so for `A[1]` it fell back to f32: a U32 element was loaded
/// with `ld.global.f32` and its bits converted as a float - 0x12345678 read
/// back as 0. `block_ptr2d_store`, the v4 forms and `atomic_add` had the same
/// lookup.
#[test]
fn a_builtin_given_an_element_address_uses_its_element_type() {
    let ptx = compiled(
        "addr_elem",
        "kernel k(A: GlobalMemory<U32>, B: GlobalMemory<U32>, C: GlobalMemory<U32>) {\n    \
         let v: U32 = block_ptr2d_load(A[1], 0, 0, 1, 1, 1);\n    \
         block_ptr2d_store(B[1], 0, 0, 1, 1, 1, v);\n    \
         atomic_add(C[1], 0, 5);\n}",
    );
    assert!(!ptx.contains(".global.f32"), "a U32 buffer must not be read or written as f32:\n{}", ptx);
    let Some(out) = run(&ptx, &[u32_bytes(&[1, 0x1234_5678, 3]), poisoned(12), u32_bytes(&[7, 9, 11])], 1)
    else {
        return skip("element address into a built-in");
    };
    assert_eq!(words(&out[1]), [P32, 0x1234_5678, P32]);
    assert_eq!(words(&out[2]), [7, 14, 11]);
}

// ───────────────────────────── no GPU needed ─────────────────────────────

/// `A[i]` as a value and `GlobalMemory::load(A[i])` are one load, emitted
/// identically - one implementation, not two that can drift.
#[test]
fn an_element_read_is_the_same_load_as_global_memory_load() {
    let body = |read: &str| {
        format!(
            "kernel k(A: GlobalMemory<F16>, Out: GlobalMemory<F32>) {{\n    \
             let v: F32 = {};\n    \
             Out[0] = v;\n}}",
            read
        )
    };
    assert_eq!(
        compiled("same_a", &body("A[1]")),
        compiled("same_b", &body("GlobalMemory::load(A[1])"))
    );
}

/// Every memory built-in still takes `A[i]` as an ADDRESS in its buffer
/// position. A position lowered as a value would load the element and use the
/// loaded bits as the address, which the backend refuses by name ("an element
/// VALUE ... is used as an ADDRESS") - so each of these compiling is the
/// statement that its position is an address. Each is also assembled where
/// `ptxas` is present, which rejects a float register used as an address
/// independently of that check.
#[test]
fn every_address_position_still_takes_an_address() {
    let probes: &[(&str, &str)] = &[
        ("store", "store(A[1], 2.0);"),
        ("assign", "A[1] = 2.0;"),
        ("gm_load", "let v: F32 = GlobalMemory::load(A[1]);\n    store(B, v);"),
        ("ptr2d_load", "let v: F32 = block_ptr2d_load(A[1], 0, 0, 1, 1, 1);\n    store(B, v);"),
        ("ptr2d_store", "block_ptr2d_store(A[1], 0, 0, 1, 1, 1, 2.0);"),
        ("ptr3d_load", "let v: F32 = block_ptr3d_load(A[1], 0, 0, 0);\n    store(B, v);"),
        ("tile_load", "let v: F32 = block_tile_load(A[1], N, 128);\n    store(B, v);"),
        ("tile_store", "block_tile_store(A[1], N, 2.0);"),
        ("bt_load", "let v: F32 = BlockTile::load(A[1], 0, N);\n    store(B, v);"),
        ("ld_v4", "let v: F32 = ld_global_v4_f32(A[1]);\n    store(B, v);"),
        ("st_v4", "st_global_v4_f32(A[1], 2.0);"),
        ("vec_add", "vec_add_v4(A[1], B[1], A[2]);"),
        ("atomic", "atomic_add(C[1], 0, 1);"),
    ];
    let mut failures = Vec::new();
    for (tag, body) in probes {
        let e = emit(
            tag,
            &format!(
                "kernel k(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<U32>, N: I32) {{\n    {}\n}}",
                body
            ),
        );
        match (&e.ptx, e.ok) {
            (Some(ptx), true) => {
                if let Err(why) = assembles(tag, ptx) {
                    failures.push(format!("{}: ptxas rejects it: {}", tag, why.trim()));
                }
            }
            _ => failures.push(format!("{}: {}", tag, e.log.trim())),
        }
    }
    assert!(failures.is_empty(), "address positions that no longer take an address:\n{}", failures.join("\n"));
}

/// `ptxas` over `ptx` at the module's own `.target`; `Ok` when there is no
/// `ptxas` to ask (the element-value check still ran in the compiler).
fn assembles(tag: &str, ptx: &str) -> Result<(), String> {
    if Command::new("ptxas").arg("--version").output().is_err() {
        return Ok(());
    }
    let target = ptx
        .lines()
        .find_map(|l| l.trim().strip_prefix(".target "))
        .expect("module declares no .target")
        .trim()
        .to_string();
    let f = std::env::temp_dir().join(format!("y_element_reads_asm_{}_{}.ptx", std::process::id(), tag));
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

/// Whether the first instruction matching `op` comes after `bar.sync`.
fn after_barrier(ptx: &str, op: &str) -> bool {
    let body = instructions(ptx);
    let bar = body.iter().position(|l| l.starts_with("bar.sync")).expect("no bar.sync");
    let at = body.iter().position(|l| l.starts_with(op)).unwrap_or_else(|| panic!("no {}:\n{}", op, ptx));
    at > bar
}

const BARRIER_PRELUDE: &str = "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>) {\n    \
     let t: U32 = thread_idx_x();\n    \
     let a: F32 = 1.0;\n    \
     barrier_sync();\n    ";

/// Memory stays on its side of a barrier: an element read, a
/// `GlobalMemory::load` in arithmetic, and an element store. HEAD moved all
/// three above `bar.sync` and called each an "independent ALU instruction".
#[test]
fn memory_is_not_moved_across_a_barrier() {
    for (tag, stmt, op) in [
        ("hoist_read", "let x: F32 = A[t] + a;\n    Out[t] = x;", "ld.global"),
        ("hoist_gm_load", "let x: F32 = GlobalMemory::load(A[t]) + a;\n    Out[t] = x;", "ld.global"),
        ("hoist_store", "Out[t] = a + a;", "st.global"),
    ] {
        let ptx = compiled(tag, &format!("{}{}\n}}", BARRIER_PRELUDE, stmt));
        assert!(after_barrier(&ptx, op), "{}: `{}` was moved above the barrier:\n{}", tag, op, ptx);
    }
}

/// The control: register arithmetic is still hoisted - "move nothing" would
/// satisfy the test above.
#[test]
fn register_arithmetic_is_still_hoisted_across_a_barrier() {
    let ptx = compiled("hoist_alu", &format!("{}let y: F32 = a * a;\n    Out[t] = y;\n}}", BARRIER_PRELUDE));
    assert!(ptx.contains("Hoisted 1 independent ALU"), "{}", ptx);
    assert!(!after_barrier(&ptx, "mul.f32"), "a * a must be hoisted above the barrier:\n{}", ptx);
}
