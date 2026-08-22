//! Derive and validate the per-lane fragment layout of
//! `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`.
//!
//! **This file exists before any emitter code, on purpose.** The FP8 path in
//! this repo required exactly this step and the deleted WGMMA surface skipped
//! it — sixteen `emit_*` methods that produced PTX the assembler rejected, and
//! two that assembled into garbage under a success message. A tensor-core
//! fragment layout is not something to read off a table and hope: a wrong
//! layout **assembles, launches, and returns plausible numbers**, so the only
//! thing that settles it is a matmul on the device compared against a CPU
//! reference.
//!
//! ## The layout, as derived from the PTX ISA
//!
//! Per warp, for `m16n8k32` with 8-bit multiplicands and a 32-bit accumulator:
//!
//! | fragment | shape | per lane | registers |
//! |---|---|---|---|
//! | A | 16x32 int8, row-major | 16 int8 | 4 x `.b32` |
//! | B | 32x8 int8, col-major | 8 int8 | 2 x `.b32` |
//! | C/D | 16x8 int32 | 4 int32 | 4 x `.s32` |
//!
//! With `g = laneid >> 2` (0..7) and `t = laneid & 3` (0..3):
//!
//! - **A** register `i` holds four *contiguous* columns, so it is one 32-bit
//!   load: `a0` at `(row g, col 4t)`, `a1` at `(g+8, 4t)`, `a2` at
//!   `(g, 4t+16)`, `a3` at `(g+8, 4t+16)`.
//! - **B** is column-major, so element `(k, n)` sits at `n*32 + k` and a lane's
//!   four k-values are contiguous too: `b0` at `(k=4t, n=g)`, `b1` at
//!   `(k=4t+16, n=g)`. Note this makes B's byte offset the *same expression* as
//!   A's, which is a coincidence of the two shapes and not a shared rule.
//! - **D** holds two contiguous columns per register pair: `d0,d1` at
//!   `(row g, col 2t)` and `d2,d3` at `(row g+8, col 2t)`.
//!
//! The test below encodes exactly that and checks it against a plain integer
//! matmul. If any of it is wrong the product is wrong, and the failure names
//! which element disagreed.
//!
//! Requires a CUDA driver; skipped with a notice otherwise.
//!
//! Run with:  cargo test --release --test ptx_int8_mma_layout -- --nocapture

/// One warp, one `m16n8k32` tile. Hand-written so the layout under test is
/// visible rather than generated.
const PROBE: &str = r#"
.version 8.4
.target sm_89
.address_size 64

.visible .entry mma_s8_probe(
    .param .u64 pA,
    .param .u64 pB,
    .param .u64 pD
)
{
    .reg .b64 %rd<20>;
    .reg .b32 %r<40>;

    ld.param.u64 %rd1, [pA];
    ld.param.u64 %rd2, [pB];
    ld.param.u64 %rd3, [pD];
    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    cvta.to.global.u64 %rd6, %rd3;

    mov.u32 %r1, %laneid;
    shr.u32 %r2, %r1, 2;        // g = laneid >> 2
    and.b32 %r3, %r1, 3;        // t = laneid & 3

    // A: row-major 16x32, byte offset = row*32 + col
    mul.lo.u32 %r4, %r2, 32;
    mul.lo.u32 %r5, %r3, 4;
    add.u32    %r6, %r4, %r5;
    cvt.u64.u32 %rd7, %r6;
    add.u64 %rd8, %rd4, %rd7;
    ld.global.u32 %r10, [%rd8];          // (g,   4t)
    add.u64 %rd9,  %rd8, 256;
    ld.global.u32 %r11, [%rd9];          // (g+8, 4t)
    add.u64 %rd10, %rd8, 16;
    ld.global.u32 %r12, [%rd10];         // (g,   4t+16)
    add.u64 %rd11, %rd8, 272;
    ld.global.u32 %r13, [%rd11];         // (g+8, 4t+16)

    // B: col-major 32x8, byte offset = n*32 + k
    cvt.u64.u32 %rd12, %r6;
    add.u64 %rd13, %rd5, %rd12;
    ld.global.u32 %r14, [%rd13];         // (k=4t,    n=g)
    add.u64 %rd14, %rd13, 16;
    ld.global.u32 %r15, [%rd14];         // (k=4t+16, n=g)

    mov.u32 %r20, 0;
    mov.u32 %r21, 0;
    mov.u32 %r22, 0;
    mov.u32 %r23, 0;

    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r24, %r25, %r26, %r27},
        {%r10, %r11, %r12, %r13},
        {%r14, %r15},
        {%r20, %r21, %r22, %r23};

    // D: row-major 16x8 int32, byte offset = (row*8 + col)*4
    mul.lo.u32 %r30, %r2, 8;
    mul.lo.u32 %r31, %r3, 2;
    add.u32 %r32, %r30, %r31;
    mul.lo.u32 %r33, %r32, 4;
    cvt.u64.u32 %rd15, %r33;
    add.u64 %rd16, %rd6, %rd15;
    st.global.v2.u32 [%rd16], {%r24, %r25};   // (g,   2t), (g,   2t+1)
    add.u64 %rd17, %rd16, 256;
    st.global.v2.u32 [%rd17], {%r26, %r27};   // (g+8, 2t), (g+8, 2t+1)

    ret;
}
"#;

const M: usize = 16;
const N: usize = 8;
const K: usize = 32;

#[test]
fn the_int8_mma_fragment_layout_is_what_the_isa_says() {
    use y::cuda_runtime::CudaContext;

    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — the int8 mma layout was not validated.");
        return;
    };

    let module = match ctx.load_ptx(PROBE, "mma_s8_probe") {
        Ok(m) => m,
        Err(e) => panic!("the int8 mma probe failed to load: {e}"),
    };

    // Values chosen so a transposed or mis-strided read cannot coincidentally
    // agree: every element is distinct modulo the ranges involved, and both
    // signs are represented so a `u8` misreading of `s8` shows up.
    let a: Vec<i8> = (0..M * K)
        .map(|i| (((i * 37 + 11) % 251) as i32 - 125) as i8)
        .collect();
    // B is COLUMN-major: b[n * K + k].
    let b: Vec<i8> = (0..K * N)
        .map(|i| (((i * 53 + 7) % 251) as i32 - 125) as i8)
        .collect();

    let d_a = ctx.alloc(M * K).unwrap();
    let d_b = ctx.alloc(K * N).unwrap();
    let d_d = ctx.alloc(M * N * 4).unwrap();
    ctx.memcpy_htod_at(&d_a, 0, &a.iter().map(|&v| v as u8).collect::<Vec<u8>>())
        .unwrap();
    ctx.memcpy_htod_at(&d_b, 0, &b.iter().map(|&v| v as u8).collect::<Vec<u8>>())
        .unwrap();
    // Poison, so a kernel that writes nothing fails rather than matching zeros.
    ctx.memset_u8(&d_d, 0xA5).unwrap();

    let args = vec![d_a.device_ptr(), d_b.device_ptr(), d_d.device_ptr()];
    ctx.launch(&module, (1, 1, 1), (32, 1, 1), 0, &args)
        .expect("launch failed");
    ctx.synchronize().expect("kernel did not complete");

    let mut raw = vec![0u8; M * N * 4];
    ctx.memcpy_dtoh_at(&mut raw, &d_d, 0).unwrap();
    let got: Vec<i32> = (0..M * N)
        .map(|i| i32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]))
        .collect();

    // D = A * B, with A row-major [M][K] and B column-major [N][K].
    let mut want = vec![0i32; M * N];
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0i32;
            for k in 0..K {
                acc += a[m * K + k] as i32 * b[n * K + k] as i32;
            }
            want[m * N + n] = acc;
        }
    }

    let mut bad = 0usize;
    let mut first = None;
    for i in 0..M * N {
        if got[i] != want[i] {
            bad += 1;
            if first.is_none() {
                first = Some(format!(
                    "D[{}][{}]: GPU {}, expected {}",
                    i / N,
                    i % N,
                    got[i],
                    want[i]
                ));
            }
        }
    }
    assert_eq!(
        bad,
        0,
        "the derived fragment layout is wrong on {bad} of {} elements — a wrong \
         layout assembles and runs, so this is the only thing that catches it. \
         First: {}",
        M * N,
        first.unwrap_or_default()
    );
}

/// The emitted int8 GEMM kernel, run on the device against a CPU reference.
///
/// The layout test above validates one `mma` in isolation; this validates the
/// **emitter path** that tiles it over a grid and loops over K. They fail
/// differently: a layout error is wrong everywhere, while a grid or K-stepping
/// error is wrong only in the tiles that moved — so the shape is chosen with
/// several tiles in both dimensions and several K steps.
#[test]
fn the_emitted_int8_gemm_matches_a_cpu_reference() {
    use std::path::Path;
    use std::process::Command;
    use y::cuda_runtime::CudaContext;

    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — the int8 GEMM was emitted but not executed.");
        return;
    };
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    let out = Command::new(bin.join("Y"))
        .arg(repo.join("tests/int8_gemm.ysu"))
        .arg("--emit-ptx")
        .current_dir(repo)
        .output()
        .expect("run Y");
    assert!(
        out.status.success(),
        "int8_gemm.ysu did not compile:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read_to_string(repo.join("tests/int8_gemm.ptx")).expect("no .ptx");
    let module = ctx.load_ptx(&ptx, "int8_gemm").expect("PTX failed to load");

    // Must match the @tile in the fixture.
    const MM: usize = 64;
    const NN: usize = 32;
    const KK: usize = 128;

    let a: Vec<i8> = (0..MM * KK)
        .map(|i| (((i * 31 + 5) % 251) as i32 - 125) as i8)
        .collect();
    let b: Vec<i8> = (0..NN * KK)
        .map(|i| (((i * 67 + 13) % 251) as i32 - 125) as i8)
        .collect();

    let d_a = ctx.alloc(MM * KK).unwrap();
    let d_b = ctx.alloc(NN * KK).unwrap();
    let d_c = ctx.alloc(MM * NN * 4).unwrap();
    ctx.memcpy_htod_at(&d_a, 0, &a.iter().map(|&v| v as u8).collect::<Vec<u8>>())
        .unwrap();
    ctx.memcpy_htod_at(&d_b, 0, &b.iter().map(|&v| v as u8).collect::<Vec<u8>>())
        .unwrap();
    // Zero, not poison: with split-K the kernel REDUCES into C
    // (`red.global.add.s32`) rather than overwriting it. A kernel that writes
    // nothing still fails, because every expected value here is non-zero.
    ctx.memset_u8(&d_c, 0).unwrap();

    let args = vec![d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr()];
    ctx.launch(
        &module,
        ((NN / 8) as u32, (MM / 16) as u32, 1),
        (32, 1, 1),
        0,
        &args,
    )
    .expect("launch failed");
    ctx.synchronize().expect("kernel did not complete");

    let mut raw = vec![0u8; MM * NN * 4];
    ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
    let got: Vec<i32> = (0..MM * NN)
        .map(|i| i32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]))
        .collect();

    let mut bad = 0usize;
    let mut first = None;
    for m in 0..MM {
        for n in 0..NN {
            let want: i32 = (0..KK)
                .map(|k| a[m * KK + k] as i32 * b[n * KK + k] as i32)
                .sum();
            if got[m * NN + n] != want {
                bad += 1;
                if first.is_none() {
                    first = Some(format!(
                        "C[{m}][{n}]: GPU {}, expected {want}",
                        got[m * NN + n]
                    ));
                }
            }
        }
    }
    assert_eq!(
        bad,
        0,
        "the emitted int8 GEMM disagrees on {bad} of {} elements. First: {}",
        MM * NN,
        first.unwrap_or_default()
    );
}
