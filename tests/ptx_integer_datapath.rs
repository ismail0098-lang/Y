//! The PTX backend's integer datapath.
//!
//! It did not have one, and it did not say so. A kernel declared
//! `GlobalMemory<U32>` with `let s: U32 = a + b;` compiled clean, assembled
//! clean under `ptxas -arch=sm_89`, printed "PTX Assembly generated
//! successfully!", exited 0 — and emitted `ld.global.f32`, `add.f32`,
//! `st.global.f32`. `U64`, `I32` and `I64` behaved identically. Every value
//! above 2^24 was silently rounded; a 254-bit field limb was destroyed
//! outright.
//!
//! That is a wrong answer, not a missing feature, and it is worse than the
//! `tma_load` / `wgmma_async` intrinsics deleted in CLAUDE.md gotcha #8 —
//! those at least produced PTX `ptxas` rejected. This one assembles, launches,
//! and returns numbers. **No assembly gate can catch it**, which is why the
//! centrepiece of this file is a differential run on the real GPU rather than
//! another substring assertion.
//!
//! The other half of the story is what is still refused: element types
//! narrower than 32 bits. An element type is also a *stride*, and every
//! address computation in `ptx_emitter.rs` shifts by log2 of the element size
//! — supporting `U8` at a 4-byte stride would read every fourth element and
//! call it the array. Refusing names the gap; guessing hides it.
//!
//! Driven through the real binary, because the refusals are build failures.

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

fn scratch() -> PathBuf {
    let d = std::env::temp_dir().join("y_ptx_int_datapath");
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Compiles `src` through the real binary. Returns (succeeded, output, ptx path).
fn compile(src: &str, name: &str) -> (bool, String, PathBuf) {
    let path = scratch().join(format!("{}.ysu", name));
    std::fs::write(&path, src).unwrap();
    let ptx = path.with_extension("ptx");
    let _ = std::fs::remove_file(&ptx);
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
    (out.status.success(), text, ptx)
}

const KERNEL: &str = r#"
kernel int_probe(A: GlobalMemory<TY>, B: GlobalMemory<TY>, C: GlobalMemory<TY>, N: I32) {
    let i: I32 = block_idx_x() * 256 + thread_idx_x();
    let a: TY = block_ptr2d_load(A, 0, i, N, 1, N);
    let b: TY = block_ptr2d_load(B, 0, i, N, 1, N);
    let s: TY = a + b;
    block_ptr2d_store(C, 0, i, N, 1, N, s);
}
fn main() {}
"#;

/// A stride is not a suffix. These stay refused until the address computations
/// carry a byte width, and the refusal has to name the reason.
#[test]
fn sub_word_element_types_are_refused() {
    for ty in ["U8", "U16", "I8", "I16"] {
        let (ok, out, ptx) = compile(&KERNEL.replace("TY", ty), &format!("sub_{}", ty));
        assert!(!ok, "GlobalMemory<{}> compiled; sub-word strides are not implemented", ty);
        assert!(
            out.contains("4- or 8-byte stride"),
            "GlobalMemory<{}> failed without naming the reason:\n{}",
            ty,
            out
        );
        assert!(!ptx.exists(), "GlobalMemory<{}> was refused but a .ptx was written", ty);
    }
}

/// The bug this file was opened for: a `U32` kernel must not be lowered as f32.
#[test]
fn integer_kernels_emit_the_integer_datapath() {
    let (ok, out, ptx) = compile(&KERNEL.replace("TY", "U32"), "int_u32");
    assert!(ok, "GlobalMemory<U32> must compile:\n{}", out);
    let text = std::fs::read_to_string(&ptx).unwrap();
    for want in ["ld.global.u32", "add.u32", "st.global.u32"] {
        assert!(text.contains(want), "integer kernel does not emit `{}`:\n{}", want, text);
    }
    // The whole point. A float instruction anywhere in an all-integer kernel
    // means some path is still hardcoded.
    for forbidden in ["ld.global.f32", "add.f32", "st.global.f32", "mov.f32"] {
        assert!(
            !text.contains(forbidden),
            "integer kernel still routes through the float datapath (`{}`):\n{}",
            forbidden,
            text
        );
    }
}

/// Signedness is not decoration: `div`, `rem`, `shr` and every comparison
/// differ between `.u32` and `.s32`, and an untyped index must stay signed.
#[test]
fn signedness_follows_the_declared_type() {
    let (ok, out, ptx) = compile(&KERNEL.replace("TY", "I32"), "int_i32");
    assert!(ok, "GlobalMemory<I32> must compile:\n{}", out);
    let text = std::fs::read_to_string(&ptx).unwrap();
    assert!(text.contains("add.s32"), "an I32 add must be signed:\n{}", text);
    // Loads and stores have no signedness, so they stay `.u32`.
    assert!(text.contains("ld.global.u32"), "I32 kernel does not load as u32:\n{}", text);
}

/// A 64-bit element is eight bytes apart, not four. This shift used to be a
/// hardcoded `shl.b64 ..., 2` at every address site.
#[test]
fn sixty_four_bit_elements_use_an_eight_byte_stride() {
    let (ok, out, ptx) = compile(&KERNEL.replace("TY", "U64"), "int_u64");
    assert!(ok, "GlobalMemory<U64> must compile:\n{}", out);
    let text = std::fs::read_to_string(&ptx).unwrap();
    assert!(text.contains("ld.global.u64"), "U64 kernel does not load 64 bits:\n{}", text);
    assert!(text.contains("add.u64"), "U64 kernel does not add 64 bits:\n{}", text);
    assert!(
        text.contains("shl.b64") && text.lines().any(|l| l.contains("shl.b64") && l.trim_end().ends_with("3;")),
        "U64 kernel does not scale its index by 8:\n{}",
        text
    );
}

/// The control, and it matters as much as every assertion above: refusing (or
/// integer-ising) everything would satisfy them and break the compiler.
#[test]
fn float_element_types_still_compile() {
    let (ok, out, ptx) = compile(&KERNEL.replace("TY", "F32"), "flt_probe");
    assert!(ok, "GlobalMemory<F32> must still compile:\n{}", out);
    let text = std::fs::read_to_string(&ptx).unwrap();
    assert!(
        text.contains("ld.global.f32") && text.contains("add.f32"),
        "F32 kernel no longer emits the float datapath:\n{}",
        text
    );
    assert!(
        !text.contains("add.u32"),
        "F32 kernel picked up integer arithmetic:\n{}",
        text
    );
}

// ── The differential run ────────────────────────────────────────────────────

/// The reference. Plain Rust `u32`/`u64` arithmetic, which is the definition
/// the kernel is claiming to implement.
fn reference(a: u32, b: u32) -> [u32; 17] {
    let sh = b & 31;
    let (wa, wb) = (a as u64, b as u64);
    let p = wa.wrapping_mul(wb).wrapping_add(wa);
    [
        a.wrapping_add(b),
        a.wrapping_sub(b),
        a.wrapping_mul(b),
        a / b,
        a % b,
        a & b,
        a | b,
        a ^ b,
        a.wrapping_shl(sh),
        a.wrapping_shr(sh),
        (a < b) as u32,
        ((a as u64 * b as u64) >> 32) as u32,
        p as u32,
        (p >> 32) as u32,
        ((wa + wb) >> 32) as u32,
        ((a as i32 as i64 as u64) >> 32) as u32,
        ((4026531841u64) >> 32) as u32,
    ]
}

const OP_NAMES: [&str; 17] = [
    "a + b", "a - b", "a * b", "a / b", "a % b", "a & b", "a | b", "a ^ b",
    "a << (b & 31)", "a >> (b & 31)", "a < b", "hi32(a * b)",
    "lo32(a64*b64 + a64)", "hi32(a64*b64 + a64)", "hi32(a + b64)", "hi32(sext(a))", "hi32(0xf0000001 as U64)",
];

/// Splitmix64, so the inputs are reproducible and full-range. Values above
/// 2^31 are the whole point: they are what separates a signed lowering from an
/// unsigned one, and values above 2^24 are what separate integers from f32.
fn splitmix(state: &mut u64) -> u32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
}

/// Compiles `tests/ptx_integer_ops.ysu`, runs it on the real GPU, and compares
/// every element of every output against the Rust reference.
///
/// Skipped, loudly, when no CUDA driver is present — the same discipline as
/// the `ptxas` and `solc` gates elsewhere in this suite.
#[test]
fn every_integer_operator_matches_a_cpu_reference_on_the_gpu() {
    use y::cuda_runtime::CudaContext;

    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — the integer datapath was not executed, only emitted.");
        return;
    };

    let src = repo().join("tests/ptx_integer_ops.ysu");
    let out = Command::new(bin())
        .arg(&src)
        .arg("--emit-ptx")
        .current_dir(repo())
        .output()
        .expect("failed to run the Y binary");
    assert!(
        out.status.success(),
        "ptx_integer_ops.ysu did not compile:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read_to_string(repo().join("tests/ptx_integer_ops.ptx"))
        .expect("no .ptx written");

    let module = ctx
        .load_ptx(&ptx, "int_ops")
        .expect("the emitted integer PTX did not load on the device");

    const N: usize = 4096;
    let mut state = 0x5EED_1234_5678_9ABCu64;
    let a: Vec<u32> = (0..N).map(|_| splitmix(&mut state)).collect();
    // A zero divisor makes `/` and `%` undefined rather than wrong, so the
    // reference and the kernel would be comparing nothing. Everything else
    // stays full-range on purpose.
    let b: Vec<u32> = (0..N).map(|_| splitmix(&mut state).max(1)).collect();

    let bytes = N * 4;
    let d_a = ctx.alloc(bytes).unwrap();
    let d_b = ctx.alloc(bytes).unwrap();
    ctx.memcpy_htod_at(&d_a, 0, bytemuck_u32(&a)).unwrap();
    ctx.memcpy_htod_at(&d_b, 0, bytemuck_u32(&b)).unwrap();

    let outs: Vec<_> = (0..17).map(|_| ctx.alloc(bytes).unwrap()).collect();
    for o in &outs {
        // Poison the outputs, so a kernel that writes nothing at all fails
        // rather than accidentally matching a zero reference.
        ctx.memset_u8(o, 0xA5).unwrap();
    }

    let mut args = vec![d_a.device_ptr(), d_b.device_ptr()];
    args.extend(outs.iter().map(|o| o.device_ptr()));
    args.push(N as u64);

    ctx.launch(&module, ((N / 256) as u32, 1, 1), (256, 1, 1), 0, &args)
        .expect("launch failed");
    ctx.synchronize().expect("kernel did not complete");

    let mut host = vec![vec![0u32; N]; 17];
    for (k, o) in outs.iter().enumerate() {
        let mut raw = vec![0u8; bytes];
        ctx.memcpy_dtoh_at(&mut raw, o, 0).unwrap();
        for i in 0..N {
            host[k][i] = u32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
        }
    }

    let mut failures = 0usize;
    let mut first: Option<String> = None;
    for i in 0..N {
        let want = reference(a[i], b[i]);
        for k in 0..17 {
            if host[k][i] != want[k] {
                failures += 1;
                if first.is_none() {
                    first = Some(format!(
                        "{} at i={}: a={} b={} -> GPU {}, expected {}",
                        OP_NAMES[k], i, a[i], b[i], host[k][i], want[k]
                    ));
                }
            }
        }
    }
    assert_eq!(
        failures,
        0,
        "{} of {} integer results disagree with the CPU reference. First: {}",
        failures,
        N * 17,
        first.unwrap_or_default()
    );
}

fn bytemuck_u32(v: &[u32]) -> &[u8] {
    // Safe: u32 has no padding and no invalid bit patterns, and the slice is
    // only read. Avoids taking a dependency for four lines.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

// ── 128-bit vector load / store ─────────────────────────────────────────────

const V4_KERNEL: &str = r#"
kernel v4_probe(A: GlobalMemory<U32>, C: GlobalMemory<U32>, N: I32) {
    let i: I32 = block_idx_x() * 256 + thread_idx_x();
    let v: U32x4 = block_ptr2d_load_v4(A, 0, i * 4, N * 4, 1, N * 4);
    block_ptr2d_store_v4(C, 0, i * 4, N * 4, 1, N * 4, v.w, v.z, v.y, v.x);
}
fn main() {}
"#;

/// The reason this intrinsic exists is that ptxas will not merge the scalar
/// form: each `block_ptr2d_load` carries its own bounds predicate, and a
/// predicated load is not a merge candidate. So the wide instruction has to be
/// emitted directly - and it must still be ONE instruction, predicate intact.
#[test]
fn vector_loads_emit_one_wide_instruction() {
    let (ok, out, ptx) = compile(V4_KERNEL, "v4_probe");
    assert!(ok, "the v4 kernel must compile:\n{}", out);
    let text = std::fs::read_to_string(&ptx).unwrap();
    assert_eq!(
        text.matches("ld.global.v4.u32").count(),
        1,
        "expected exactly one wide load:\n{}",
        text
    );
    assert_eq!(
        text.matches("st.global.v4.u32").count(),
        1,
        "expected exactly one wide store:\n{}",
        text
    );
    // Predication is not sacrificed for width; that was never the trade.
    assert!(
        text.contains("@%p") && text.contains("and.pred"),
        "the v4 access lost its bounds predicate:\n{}",
        text
    );
    // Four lanes read back must not cost four loads.
    assert!(
        !text.contains("ld.global.u32"),
        "a scalar load survived alongside the vector one:\n{}",
        text
    );
}

/// A lane that does not exist is a refusal, not lane 0. Reading `.q` as `.x`
/// is the silent-substitution failure this repo keeps finding.
#[test]
fn a_nonexistent_lane_is_refused() {
    let src = V4_KERNEL.replace("v.w, v.z, v.y, v.x", "v.q, v.z, v.y, v.x");
    let (ok, out, _) = compile(&src, "v4_badlane");
    assert!(!ok, "`.q` on a 4-wide vector must be refused");
    assert!(
        out.contains("not one of") && out.contains(".x .y .z .w"),
        "the refusal did not name the valid lanes:\n{}",
        out
    );
}
