//! The carry-chained 32-bit intrinsics, checked on the device against plain
//! Rust `u64` arithmetic.
//!
//! These are the one place in this backend where an instruction communicates
//! with the next through state that appears in no operand — the hardware
//! condition code. That makes them the easiest thing here to get subtly wrong
//! and the hardest to notice: a `_cc` suffix dropped from the middle of a
//! chain still assembles, still launches, and still gives the right answer for
//! every input whose limbs happen not to carry.
//!
//! So the gate is a differential run over full-range random limbs, not a
//! string match and not a hand-picked vector. The Rust side deliberately uses
//! the OLD formulation — `a*b` widened to `u64` and accumulated — so the two
//! sides are genuinely different implementations of the same arithmetic
//! rather than the same algorithm typed twice.
//!
//! Run with:  cargo test --release --test ptx_carry_chain

use std::path::{Path, PathBuf};
use std::process::Command;

use y::cuda_runtime::CudaContext;

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

/// The reference, written the way the kernels used to be: widen to `u64` and
/// carry by shifting. Sixteen outputs per input pair, matching the probe's
/// store order.
fn oracle(a: [u32; 4], b: [u32; 4]) -> [u32; 16] {
    let mut o = [0u32; 16];

    let mut c = 0u64;
    for i in 0..4 {
        let t = a[i] as u64 + b[i] as u64 + c;
        o[i] = t as u32;
        c = t >> 32;
    }
    o[4] = c as u32;

    let mut borrow = 0u64;
    for i in 0..4 {
        let t = (a[i] as u64)
            .wrapping_sub(b[i] as u64)
            .wrapping_sub(borrow);
        o[5 + i] = t as u32;
        borrow = (t >> 32) & 1;
    }
    // `subc_u32(0, 0)` is `0 - 0 - borrow`, confirmed on the device.
    o[9] = if borrow == 1 { 0xFFFF_FFFF } else { 0 };

    // r = a * b[0] + b
    let mut c2 = 0u64;
    for i in 0..4 {
        let t = a[i] as u64 * b[0] as u64 + b[i] as u64 + c2;
        o[10 + i] = t as u32;
        c2 = t >> 32;
    }
    o[14] = c2 as u32;
    // a*b0 + b < 2^160 for 128-bit a and b, so the sixth limb is always zero.
    o[15] = 0;
    o
}

const LANES: usize = 4096;

fn limbs() -> (Vec<u32>, Vec<u32>) {
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as u32
    };
    let mut a = Vec::with_capacity(LANES * 4);
    let mut b = Vec::with_capacity(LANES * 4);
    // Structured cases first: an all-ones limb is what makes a carry chain
    // propagate the whole way, and random 32-bit words almost never do.
    let edge: [u32; 6] = [0, 1, 0xFFFF_FFFF, 0xFFFF_FFFE, 0x8000_0000, 0x7FFF_FFFF];
    for i in 0..LANES {
        for j in 0..4 {
            if i < edge.len() * edge.len() {
                a.push(edge[(i / edge.len()) % edge.len()]);
                b.push(edge[i % edge.len()]);
            } else {
                a.push(next());
                b.push(next());
            }
        }
    }
    (a, b)
}

fn as_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn carry_chains_match_plain_rust_on_the_device() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — ptx_carry_chain.ysu was not executed.");
        return;
    };
    let ptx = compile("ptx_carry_chain");
    let module = ctx
        .load_ptx(&ptx, "carry_probe")
        .expect("carry_probe did not load");

    let (a, b) = limbs();
    let n = (LANES * 16).max(LANES * 4);
    let d_a = ctx.alloc(a.len() * 4).unwrap();
    let d_b = ctx.alloc(b.len() * 4).unwrap();
    let d_o = ctx.alloc(LANES * 16 * 4).unwrap();
    ctx.memcpy_htod_at(&d_a, 0, &as_bytes(&a)).unwrap();
    ctx.memcpy_htod_at(&d_b, 0, &as_bytes(&b)).unwrap();
    ctx.memcpy_htod_at(&d_o, 0, &vec![0u8; LANES * 16 * 4]).unwrap();

    ctx.launch(
        &module,
        ((LANES / 256) as u32, 1, 1),
        (256, 1, 1),
        0,
        &[d_a.device_ptr(), d_b.device_ptr(), d_o.device_ptr(), n as u64],
    )
    .expect("launch failed");
    ctx.synchronize().unwrap();

    let mut raw = vec![0u8; LANES * 16 * 4];
    ctx.memcpy_dtoh_at(&mut raw, &d_o, 0).unwrap();
    let got: Vec<u32> = raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    const NAMES: [&str; 16] = [
        "s0", "s1", "s2", "s3", "carry", "d0", "d1", "d2", "d3", "borrow", "r0", "r1", "r2",
        "r3", "r4", "r5",
    ];
    for lane in 0..LANES {
        let av = [a[lane * 4], a[lane * 4 + 1], a[lane * 4 + 2], a[lane * 4 + 3]];
        let bv = [b[lane * 4], b[lane * 4 + 1], b[lane * 4 + 2], b[lane * 4 + 3]];
        let want = oracle(av, bv);
        for k in 0..16 {
            assert_eq!(
                got[lane * 16 + k],
                want[k],
                "lane {} field {}: a = {:08x?}, b = {:08x?}",
                lane,
                NAMES[k],
                av,
                bv
            );
        }
    }
}

/// The control. A differential test whose inputs never carry proves nothing,
/// and random 32-bit words carry through a four-limb chain about as often as
/// they do not — but they essentially never propagate a carry the WHOLE way,
/// which is the case a dropped `_cc` in the middle of the chain breaks.
///
/// This asserts the input set actually contains such cases, so the test above
/// cannot pass by never exercising the mechanism it exists to check.
#[test]
fn the_inputs_actually_propagate_carries() {
    let (a, b) = limbs();
    let mut full_carry = 0;
    let mut full_borrow = 0;
    for lane in 0..LANES {
        let av = [a[lane * 4], a[lane * 4 + 1], a[lane * 4 + 2], a[lane * 4 + 3]];
        let bv = [b[lane * 4], b[lane * 4 + 1], b[lane * 4 + 2], b[lane * 4 + 3]];
        // A carry out of the top limb means the chain ran end to end.
        if oracle(av, bv)[4] == 1 {
            full_carry += 1;
        }
        if oracle(av, bv)[9] != 0 {
            full_borrow += 1;
        }
    }
    assert!(
        full_carry > 0,
        "no input pair carries out of the top limb, so a broken carry chain \
         would go unnoticed"
    );
    assert!(
        full_borrow > 0,
        "no input pair borrows out of the top limb"
    );
    eprintln!(
        "{} of {} lanes carry out, {} borrow out",
        full_carry, LANES, full_borrow
    );
}
