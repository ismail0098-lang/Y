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

/// Sub-word element types carry their own stride, and their own extension.
///
/// A stride is not a suffix: the address computation must shift by
/// `log2(bytes)`, which is **0** for a byte type. If it shifts by a hardcoded
/// log2(4) the kernel reads every fourth element and reports it as the array —
/// assembles, launches, wrong answer. The extension is the second half:
/// `ld.global.s8` sign-extends and `ld.global.u8` zero-extends, and swapping
/// them turns every negative byte into a large positive one.
#[test]
fn sub_word_buffers_get_their_own_stride_and_extension() {
    // The local must be I32: a sub-word LOCAL is refused on purpose (see
    // `sub_word_locals_are_refused`), so the element type appears only on the
    // buffer, which is where a width is unambiguously a storage format.
    let k = KERNEL
        .replace("let a: TY", "let a: I32")
        .replace("let b: TY", "let b: I32")
        .replace("let s: TY", "let s: I32");
    for (ty, shift, suffix) in [
        ("U8", 0, "ld.global.u8"),
        ("I8", 0, "ld.global.s8"),
        ("U16", 1, "ld.global.u16"),
        ("I16", 1, "ld.global.s16"),
    ] {
        let src = k
            .replace("A: GlobalMemory<TY>", &format!("A: GlobalMemory<{ty}>"))
            .replace("B: GlobalMemory<TY>", &format!("B: GlobalMemory<{ty}>"))
            .replace("TY", "I32");
        let (ok, out, ptx) = compile(&src, &format!("sub_{ty}"));
        assert!(ok, "GlobalMemory<{ty}> must compile now:\n{out}");
        let text = std::fs::read_to_string(&ptx).expect("no .ptx written");
        assert!(
            text.contains(suffix),
            "GlobalMemory<{ty}> did not emit `{suffix}` — the load's extension is wrong, \
             which silently changes the value of every negative element:\n{text}"
        );
        assert!(
            text.contains(&format!(", {shift};")),
            "GlobalMemory<{ty}> did not shift its index by {shift} — the stride is wrong, \
             so the kernel is reading the wrong elements:\n{text}"
        );
    }
}

/// The sub-word datapath, executed on the real device against a Rust reference.
///
/// **The emitted-PTX assertions above cannot catch what this catches.** A wrong
/// stride and a wrong extension both produce PTX that `ptxas` accepts and the
/// GPU runs; the only symptom is the numbers. Two failure modes are targeted
/// specifically:
///
/// - **Stride.** Every element is distinct and index-dependent, so reading at a
///   4-byte stride returns element `4i` instead of `i` and disagrees almost
///   everywhere rather than subtly.
/// - **Extension.** Inputs span the full signed and unsigned byte/half ranges,
///   so `s8` vs `u8` disagree on every negative value — half the data.
#[test]
fn the_sub_word_datapath_matches_a_cpu_reference_on_the_gpu() {
    use y::cuda_runtime::CudaContext;

    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — the sub-word datapath was emitted but not executed.");
        return;
    };

    let src = repo().join("tests/ptx_subword_ops.ysu");
    let out = Command::new(bin())
        .arg(&src)
        .arg("--emit-ptx")
        .current_dir(repo())
        .output()
        .expect("failed to run the Y binary");
    assert!(
        out.status.success(),
        "ptx_subword_ops.ysu did not compile:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read_to_string(repo().join("tests/ptx_subword_ops.ptx"))
        .expect("no .ptx written");
    let module = ctx.load_ptx(&ptx, "subword_ops").expect("PTX failed to load");

    const N: usize = 4096;
    // Full range in both signednesses, and index-dependent so a stride error
    // cannot coincidentally agree.
    let a8: Vec<i8> = (0..N).map(|i| (i as i32 % 256 - 128) as i8).collect();
    let b8: Vec<u8> = (0..N).map(|i| ((i * 7 + 3) % 256) as u8).collect();
    let a16: Vec<i16> = (0..N).map(|i| ((i * 37) as i32 % 65536 - 32768) as i16).collect();
    let b16: Vec<u16> = (0..N).map(|i| ((i * 811 + 17) % 65536) as u16).collect();

    let d_a8 = ctx.alloc(N).unwrap();
    let d_b8 = ctx.alloc(N).unwrap();
    let d_a16 = ctx.alloc(N * 2).unwrap();
    let d_b16 = ctx.alloc(N * 2).unwrap();
    ctx.memcpy_htod_at(&d_a8, 0, &a8.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
    ctx.memcpy_htod_at(&d_b8, 0, &b8).unwrap();
    ctx.memcpy_htod_at(&d_a16, 0, &a16.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    ctx.memcpy_htod_at(&d_b16, 0, &b16.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();

    // Five i32 outputs, then one i8 round-trip output.
    let outs32: Vec<_> = (0..5).map(|_| ctx.alloc(N * 4).unwrap()).collect();
    let out_shr = ctx.alloc(N * 4).unwrap();
    let out8 = ctx.alloc(N).unwrap();
    for o in &outs32 {
        ctx.memset_u8(o, 0xA5).unwrap();
    }
    ctx.memset_u8(&out8, 0xA5).unwrap();
    ctx.memset_u8(&out_shr, 0xA5).unwrap();

    let mut args = vec![
        d_a8.device_ptr(),
        d_b8.device_ptr(),
        d_a16.device_ptr(),
        d_b16.device_ptr(),
    ];
    args.extend(outs32.iter().map(|o| o.device_ptr()));
    args.push(out8.device_ptr());
    args.push(out_shr.device_ptr());
    args.push(N as u64);

    ctx.launch(&module, ((N / 256) as u32, 1, 1), (256, 1, 1), 0, &args)
        .expect("launch failed");
    ctx.synchronize().expect("kernel did not complete");

    let mut got32 = vec![vec![0i32; N]; 5];
    for (k, o) in outs32.iter().enumerate() {
        let mut raw = vec![0u8; N * 4];
        ctx.memcpy_dtoh_at(&mut raw, o, 0).unwrap();
        for i in 0..N {
            got32[k][i] = i32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
        }
    }
    let mut raw8 = vec![0u8; N];
    ctx.memcpy_dtoh_at(&mut raw8, &out8, 0).unwrap();
    let mut raw_shr = vec![0u8; N * 4];
    ctx.memcpy_dtoh_at(&mut raw_shr, &out_shr, 0).unwrap();

    let names = ["s8 sign-extend", "u8 zero-extend", "s16 sign-extend", "u16 zero-extend", "32-bit sum"];
    let mut failures = 0usize;
    let mut first: Option<String> = None;
    for i in 0..N {
        let (x8, y8) = (a8[i] as i32, b8[i] as i32);
        let (x16, y16) = (a16[i] as i32, b16[i] as i32);
        // The promotion rule: arithmetic is 32-bit and does NOT wrap at 8 bits.
        let want = [x8, y8, x16, y16, x8 * y8 + x16 - y16];
        for k in 0..5 {
            if got32[k][i] != want[k] {
                failures += 1;
                if first.is_none() {
                    first = Some(format!(
                        "{} at i={}: GPU {}, expected {}",
                        names[k], i, got32[k][i], want[k]
                    ));
                }
            }
        }
        // Signedness of the promotion, observed inline. An arithmetic shift
        // of a sign-extended negative byte; unsigned would give a huge value.
        let got_shr = i32::from_le_bytes([
            raw_shr[i * 4], raw_shr[i * 4 + 1], raw_shr[i * 4 + 2], raw_shr[i * 4 + 3],
        ]);
        if got_shr != (x8 >> 2) {
            failures += 1;
            if first.is_none() {
                first = Some(format!(
                    "promoted signedness at i={}: a8={} -> GPU {}, expected {}",
                    i, x8, got_shr, x8 >> 2
                ));
            }
        }

        // The round trip: storing an i32 into an I8 buffer truncates.
        let want8 = (x8 + y8) as u8;
        if raw8[i] != want8 {
            failures += 1;
            if first.is_none() {
                first = Some(format!(
                    "i8 store truncation at i={}: GPU {}, expected {}",
                    i, raw8[i], want8
                ));
            }
        }
    }
    assert_eq!(
        failures, 0,
        "the sub-word datapath disagrees with the CPU on {failures} of {} comparisons. \
         First: {}",
        N * 7,
        first.unwrap_or_default()
    );
}

/// A sub-word LOCAL is refused, and that is a semantic boundary rather than a
/// missing feature.
///
/// PTX has no sub-word register class, so `let x: I8` would be a 32-bit value
/// whose declared type promises wraparound it will not perform. Accepting it
/// silently is the failure this file's whole design rule is about, so the
/// refusal must name the reason and write no `.ptx`.
#[test]
fn sub_word_locals_are_refused() {
    for ty in ["U8", "U16", "I8", "I16"] {
        let src = KERNEL
            .replace("A: GlobalMemory<TY>", "A: GlobalMemory<I32>")
            .replace("B: GlobalMemory<TY>", "B: GlobalMemory<I32>")
            .replace("C: GlobalMemory<TY>", "C: GlobalMemory<I32>")
            .replace("let a: TY", &format!("let a: {ty}"))
            .replace("TY", "I32");
        let (ok, out, ptx) = compile(&src, &format!("sublocal_{ty}"));
        assert!(!ok, "`let a: {ty}` compiled; a sub-word local has no register to live in");
        assert!(
            out.contains("no sub-word register class"),
            "`let a: {ty}` failed without naming the reason:\n{out}"
        );
        assert!(!ptx.exists(), "`let a: {ty}` was refused but a .ptx was written");
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
