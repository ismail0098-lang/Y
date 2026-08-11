//! BN254 field multiplication on the GPU, written in Y, checked against the
//! CPU field it has to agree with.
//!
//! `tests/bn254_fr_mul.ysu` is a CIOS Montgomery multiply over eight 32-bit
//! limbs. It is the first real cryptographic kernel this backend can express:
//! it needs `mul_wide_u32` (the high half of a limb product IS the carry),
//! 64-bit accumulation, unsigned comparison, and typed integer loads - none of
//! which existed before `tests/ptx_integer_datapath.rs` went in. A
//! `GlobalMemory<U32>` used to be loaded with `ld.global.f32`, which does not
//! merely lose precision on a field limb, it destroys it.
//!
//! The oracle is `zk_field.rs`, the same field the R1CS emitter and the
//! witness solver use, reached only through its public API - `Fr`'s limbs are
//! private on purpose, since they are in Montgomery form and reading past the
//! API gets you a different number with no type error.
//!
//! What this does NOT claim: that the kernel is fast. Its accumulator lives in
//! per-thread global scratch, because the backend has no local arrays. That is
//! the next piece of compiler work, and is called out in the kernel itself.

use std::path::{Path, PathBuf};
use std::process::Command;

use y::cuda_runtime::CudaContext;
use y::zk_field::Fr;

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

/// BN254's Fr modulus, little-endian 32-bit limbs.
const P32: [u32; 8] = [
    0xf000_0001, 0x43e1_f593, 0x79b9_7091, 0x2833_e848,
    0x8181_585d, 0xb850_45b6, 0xe131_a029, 0x3064_4e72,
];

/// `-p^-1 mod 2^32`, the CIOS reduction constant.
const N_PRIME: u32 = 0xefff_ffff;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A field element's canonical value as eight 32-bit limbs.
fn limbs32(x: &Fr) -> [u32; 8] {
    let l = x.to_limbs();
    let mut out = [0u32; 8];
    for i in 0..4 {
        out[2 * i] = l[i] as u32;
        out[2 * i + 1] = (l[i] >> 32) as u32;
    }
    out
}

/// `R mod p` as a field element, computed with nothing but the public API:
/// 256 doublings of one. The kernel consumes `x * R mod p` for its left
/// operand, and `Fr`'s Montgomery limbs are deliberately unreachable.
fn r_mod_p() -> Fr {
    let mut r = Fr::from_u64(1);
    for _ in 0..256 {
        r = r.add(&r);
    }
    r
}

fn as_bytes(v: &[u32]) -> &[u8] {
    // u32 has no padding and no invalid bit patterns; read-only view.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// Compiles a `.ysu` of the same stem and loads `entry` onto the device.
fn load_kernel(ctx: &CudaContext, entry: &str) -> y::cuda_runtime::KernelModule {
    let src = repo().join(format!("tests/{}.ysu", entry));
    let out = Command::new(bin())
        .arg(&src)
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
    // A single float instruction in a field kernel means some path is still
    // hardcoded and the limbs are being rounded.
    assert!(
        !ptx.contains("ld.global.f32") && !ptx.contains("add.f32"),
        "{} is routing limbs through the float datapath",
        entry
    );
    ctx.load_ptx(&ptx, entry)
        .unwrap_or_else(|e| panic!("{} did not load on the device: {}", entry, e))
}

fn read_u32(ctx: &CudaContext, buf: &y::cuda_runtime::DeviceBuffer, n: usize) -> Vec<u32> {
    let mut raw = vec![0u8; n * 4];
    ctx.memcpy_dtoh_at(&mut raw, buf, 0).unwrap();
    raw.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Operands: the values that break a careless implementation first (zero, one,
/// p-1 where the conditional subtract must fire, and R), then the full field.
fn operands(n: usize) -> (Vec<Fr>, Vec<Fr>) {
    let r = r_mod_p();
    let pm1 = Fr::from_u64(0).sub(&Fr::from_u64(1));
    let mut xs: Vec<Fr> = vec![Fr::from_u64(0), Fr::from_u64(1), pm1, r];
    let mut ys: Vec<Fr> = vec![pm1, pm1, pm1, Fr::from_u64(1)];
    let mut state = 0xF1E1_D1C1_B1A1_9181u64;
    while xs.len() < n {
        xs.push(Fr::from_limbs_reduce([
            splitmix(&mut state), splitmix(&mut state),
            splitmix(&mut state), splitmix(&mut state),
        ]));
        ys.push(Fr::from_limbs_reduce([
            splitmix(&mut state), splitmix(&mut state),
            splitmix(&mut state), splitmix(&mut state),
        ]));
    }
    (xs, ys)
}

/// The whole point: a field multiply computed on the GPU by Y-generated PTX
/// must equal the one `zk_emitter` and `zk_witness` compute on the CPU.
///
/// Skipped, loudly, without a CUDA driver — the same discipline as the `ptxas`
/// and `solc` gates elsewhere in this suite.
#[test]
fn gpu_montgomery_multiply_agrees_with_the_cpu_field() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!(
            "SKIP: no CUDA driver — bn254_fr_mul.ysu was not executed. \
             It still compiles and assembles; nothing here checked its arithmetic."
        );
        return;
    };

    let out = Command::new(bin())
        .arg(repo().join("tests/bn254_fr_mul.ysu"))
        .arg("--emit-ptx")
        .current_dir(repo())
        .output()
        .expect("failed to run the Y binary");
    assert!(
        out.status.success(),
        "bn254_fr_mul.ysu did not compile:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read_to_string(repo().join("tests/bn254_fr_mul.ptx"))
        .expect("no .ptx written");
    // If a single float instruction survives in a field kernel, some path is
    // still hardcoded and the limbs are being rounded.
    assert!(
        !ptx.contains("ld.global.f32") && !ptx.contains("add.f32"),
        "the field kernel is routing limbs through the float datapath"
    );

    let module = ctx
        .load_ptx(&ptx, "bn254_fr_mul")
        .expect("the emitted field PTX did not load on the device");

    const N: usize = 4096;
    let r = r_mod_p();
    let mut state = 0xF1E1_D1C1_B1A1_9181u64;

    // Operands drawn from the full field, plus a hand-picked prefix of the
    // values that break a careless implementation: zero, one, p-1 (the largest
    // element, where the conditional subtract must fire), and R itself.
    let mut xs: Vec<Fr> = vec![
        Fr::from_u64(0),
        Fr::from_u64(1),
        Fr::from_u64(0).sub(&Fr::from_u64(1)),
        r,
    ];
    let mut ys: Vec<Fr> = vec![
        Fr::from_u64(0).sub(&Fr::from_u64(1)),
        Fr::from_u64(0).sub(&Fr::from_u64(1)),
        Fr::from_u64(0).sub(&Fr::from_u64(1)),
        Fr::from_u64(1),
    ];
    while xs.len() < N {
        let a = Fr::from_limbs_reduce([
            splitmix(&mut state), splitmix(&mut state),
            splitmix(&mut state), splitmix(&mut state),
        ]);
        let b = Fr::from_limbs_reduce([
            splitmix(&mut state), splitmix(&mut state),
            splitmix(&mut state), splitmix(&mut state),
        ]);
        xs.push(a);
        ys.push(b);
    }

    // The kernel computes `A * B * R^-1 mod p`, so the left operand goes in as
    // `x * R mod p` and the result comes out canonical.
    let mut a_host = Vec::with_capacity(N * 8);
    let mut b_host = Vec::with_capacity(N * 8);
    let mut expect = Vec::with_capacity(N * 8);
    for i in 0..N {
        a_host.extend_from_slice(&limbs32(&xs[i].mul(&r)));
        b_host.extend_from_slice(&limbs32(&ys[i]));
        expect.extend_from_slice(&limbs32(&xs[i].mul(&ys[i])));
    }

    let d_a = ctx.alloc(N * 8 * 4).unwrap();
    let d_b = ctx.alloc(N * 8 * 4).unwrap();
    let d_p = ctx.alloc(8 * 4).unwrap();
    let d_t = ctx.alloc(N * 18 * 4).unwrap();
    let d_out = ctx.alloc(N * 8 * 4).unwrap();
    ctx.memcpy_htod_at(&d_a, 0, as_bytes(&a_host)).unwrap();
    ctx.memcpy_htod_at(&d_b, 0, as_bytes(&b_host)).unwrap();
    ctx.memcpy_htod_at(&d_p, 0, as_bytes(&P32)).unwrap();
    // Poison the output, so a kernel that writes nothing cannot pass by
    // matching a zero expectation.
    ctx.memset_u8(&d_out, 0xA5).unwrap();

    let args = vec![
        d_a.device_ptr(),
        d_b.device_ptr(),
        d_p.device_ptr(),
        d_t.device_ptr(),
        d_out.device_ptr(),
        N_PRIME as u64,
        N as u64,
    ];
    ctx.launch(&module, ((N / 256) as u32, 1, 1), (256, 1, 1), 0, &args)
        .expect("launch failed");
    ctx.synchronize().expect("kernel did not complete");

    let mut raw = vec![0u8; N * 8 * 4];
    ctx.memcpy_dtoh_at(&mut raw, &d_out, 0).unwrap();
    let got: Vec<u32> = raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut wrong = 0usize;
    let mut first = String::new();
    for i in 0..N {
        let g = &got[i * 8..i * 8 + 8];
        let e = &expect[i * 8..i * 8 + 8];
        if g != e {
            wrong += 1;
            if first.is_empty() {
                first = format!(
                    "element {}: x={} y={}\n  GPU      {:08x?}\n  zk_field {:08x?}",
                    i,
                    xs[i].to_decimal_string(),
                    ys[i].to_decimal_string(),
                    g,
                    e
                );
            }
        }
    }
    assert_eq!(wrong, 0, "{} of {} products disagree.\n{}", wrong, N, first);
}

/// The same check against the register-resident kernel.
///
/// `tests/bn254_fr_mul_fast.ysu` is generated by `tools/gen_bn254_kernels.py`:
/// the same CIOS algorithm with every limb in a register and the modulus as
/// immediates, unrolled because the backend has no local arrays and a loop
/// over limb indices therefore cannot be expressed. It must agree with
/// `zk_field` exactly as the rolled version does - a faster kernel that is
/// only nearly right is worth nothing.
#[test]
fn the_register_resident_kernel_agrees_too() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — bn254_fr_mul_fast.ysu was not executed.");
        return;
    };
    let module = load_kernel(&ctx, "bn254_fr_mul_fast");

    const N: usize = 4096;
    let (xs, ys) = operands(N);
    let r = r_mod_p();
    let mut a_host = Vec::with_capacity(N * 8);
    let mut b_host = Vec::with_capacity(N * 8);
    let mut expect = Vec::with_capacity(N * 8);
    for i in 0..N {
        a_host.extend_from_slice(&limbs32(&xs[i].mul(&r)));
        b_host.extend_from_slice(&limbs32(&ys[i]));
        expect.extend_from_slice(&limbs32(&xs[i].mul(&ys[i])));
    }

    let d_a = ctx.alloc(N * 8 * 4).unwrap();
    let d_b = ctx.alloc(N * 8 * 4).unwrap();
    let d_out = ctx.alloc(N * 8 * 4).unwrap();
    ctx.memcpy_htod_at(&d_a, 0, as_bytes(&a_host)).unwrap();
    ctx.memcpy_htod_at(&d_b, 0, as_bytes(&b_host)).unwrap();
    ctx.memset_u8(&d_out, 0xA5).unwrap();

    let args = vec![d_a.device_ptr(), d_b.device_ptr(), d_out.device_ptr(), N as u64];
    ctx.launch(&module, ((N / 256) as u32, 1, 1), (256, 1, 1), 0, &args).unwrap();
    ctx.synchronize().unwrap();

    let got = read_u32(&ctx, &d_out, N * 8);
    for i in 0..N {
        assert_eq!(
            &got[i * 8..i * 8 + 8],
            &expect[i * 8..i * 8 + 8],
            "element {}: x={} y={}", i, xs[i].to_decimal_string(), ys[i].to_decimal_string()
        );
    }
}

/// What the kernel costs, next to the CPU field it agrees with.
///
/// Deliberately `--ignored`: it is a measurement, not a property. Run with
/// `cargo test --release --features zk --test zk_gpu_field -- --ignored --nocapture`.
///
/// The number to expect is *bad*, and the reason is structural rather than
/// tuning: the eight-limb accumulator lives in per-thread global scratch, so
/// every one of the ~128 multiply-accumulate steps in a CIOS pass pays a
/// global load and a global store. A register-resident version needs indexable
/// local storage in the backend. Recording the figure here is what makes that
/// claim checkable rather than an excuse.
#[test]
#[ignore]
fn what_the_gpu_field_multiply_costs() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let out = Command::new(bin())
        .arg(repo().join("tests/bn254_fr_mul.ysu"))
        .arg("--emit-ptx")
        .current_dir(repo())
        .output()
        .unwrap();
    assert!(out.status.success());
    let ptx = std::fs::read_to_string(repo().join("tests/bn254_fr_mul.ptx")).unwrap();
    let module = ctx.load_ptx(&ptx, "bn254_fr_mul").unwrap();

    const N: usize = 1 << 20;
    let d_a = ctx.alloc(N * 8 * 4).unwrap();
    let d_b = ctx.alloc(N * 8 * 4).unwrap();
    let d_p = ctx.alloc(8 * 4).unwrap();
    let d_t = ctx.alloc(N * 18 * 4).unwrap();
    let d_out = ctx.alloc(N * 8 * 4).unwrap();
    ctx.memcpy_htod_at(&d_p, 0, as_bytes(&P32)).unwrap();
    ctx.memset_u8(&d_a, 0x11).unwrap();
    ctx.memset_u8(&d_b, 0x22).unwrap();

    let args = vec![
        d_a.device_ptr(), d_b.device_ptr(), d_p.device_ptr(),
        d_t.device_ptr(), d_out.device_ptr(), N_PRIME as u64, N as u64,
    ];
    let us = ctx
        .time_launches(&module, ((N / 256) as u32, 1, 1), (256, 1, 1), 0, &[args], 20)
        .unwrap();
    let per_mul_ns = us * 1000.0 / N as f64;

    // The register-resident kernel, same shape, same operands.
    let fast = load_kernel(&ctx, "bn254_fr_mul_fast");
    let fargs = vec![d_a.device_ptr(), d_b.device_ptr(), d_out.device_ptr(), N as u64];
    let fus = ctx
        .time_launches(&fast, ((N / 256) as u32, 1, 1), (256, 1, 1), 0, &[fargs], 50)
        .unwrap();
    let fast_ns = fus * 1000.0 / N as f64;

    // The CPU side of the same operation, same machine, single-threaded.
    let r = r_mod_p();
    let x = Fr::from_u64(0).sub(&Fr::from_u64(7)).mul(&r);
    let mut acc = Fr::from_u64(3);
    let reps = 2_000_000u32;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        acc = acc.mul(&x);
    }
    let cpu_ns = t0.elapsed().as_secs_f64() * 1e9 / reps as f64;
    std::hint::black_box(acc);

    println!("\nBN254 Fr multiply, {} elements", N);
    println!("  GPU (Y, global scratch):   {:>8.3} ms = {:>7.3} ns/mul ({:>8.1} M mul/s)",
             us / 1000.0, per_mul_ns, 1000.0 / per_mul_ns);
    println!("  GPU (Y, registers):        {:>8.3} ms = {:>7.3} ns/mul ({:>8.1} M mul/s)",
             fus / 1000.0, fast_ns, 1000.0 / fast_ns);
    println!("  CPU (zk_field, 1 thread):  {:>19.3} ns/mul", cpu_ns);
    println!();
    println!("  registers vs global scratch: {:.1}x", per_mul_ns / fast_ns);
    println!("  registers vs one CPU core:   {:.1}x", cpu_ns / fast_ns);
}

/// The control. Every assertion above is an equality against `zk_field`, so a
/// kernel that returned its own input would still have to be caught by
/// something — this checks the comparison can fail at all, by asking for a
/// product the GPU was never given.
#[test]
fn the_reference_is_not_vacuous() {
    let r = r_mod_p();
    let two = Fr::from_u64(2);
    let three = Fr::from_u64(3);
    assert_eq!(two.mul(&three).to_decimal_string(), "6");
    // R is not 1, so the Montgomery staging above is doing something.
    assert_ne!(r.to_decimal_string(), "1");
    assert_ne!(limbs32(&two.mul(&three)), limbs32(&two));
}
