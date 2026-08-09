//! What ZK emission time is actually made of.
//!
//! The obvious assumption — that a circuit dominated by field arithmetic is
//! dominated by the *cost of a field multiply* — was measurably wrong, and
//! believing it would have sent the optimisation at the wrong target. On a chain
//! of 1000 Poseidon hashes (241,000 constraints) the emitter performs 4.12 M
//! `Fr::mul` and 7.81 M `Fr::add`; when `Fr` was a heap `BigUint` those cost
//! 280 ns and 80 ns, which is only ~1.8 s of an 8.6 s emit. The other 6.8 s was
//! **356 M allocations**. `Fr` is now a `Copy` `[u64; 4]` in Montgomery form and
//! allocates nothing; see `docs/zk_emit_profile.md`.
//!
//! This file stays because the attribution is what has to be re-derived after
//! any change to the representation, not the conclusion. Two of its assertions
//! are permanent: the operands must be full-width, and the field primitives must
//! not allocate.
//!
//! Run with:
//!   cargo test --release --features zk --test zk_field_microbench -- --ignored --nocapture

#![cfg(feature = "zk")]

use std::time::Instant;
use y::zk_emitter::{Fr, LinearCombination};

/// Test-local counting allocator, so the split between "allocations inside
/// `Fr::mul`" and "allocations in the emitter's own bookkeeping" can be
/// measured. The compiler binary has its own copy; this one only covers this
/// test process.
mod ca {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static ALLOCS: AtomicU64 = AtomicU64::new(0);
    pub struct C;
    unsafe impl GlobalAlloc for C {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            ALLOCS.fetch_add(1, Relaxed);
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            ALLOCS.fetch_add(1, Relaxed);
            unsafe { System.realloc(p, l, n) }
        }
    }
    pub fn n() -> u64 {
        ALLOCS.load(Relaxed)
    }
}

#[global_allocator]
static A: ca::C = ca::C;

/// Field elements spread across the full 254-bit range.
///
/// Small operands understate the cost. This mattered enormously under the old
/// `BigUint`, which trimmed leading zero limbs, so `Fr(3) * Fr(5)` multiplied
/// one limb by one limb and skipped the reduction entirely — a benchmark built
/// from small constants measured a function the circuit never called. Montgomery
/// multiplication is constant-width and so immune to that particular trap, but
/// the operands stay full-width anyway: the emitter's real coefficients are
/// Poseidon round keys, and a benchmark should use the inputs the thing under
/// test actually sees.
fn spread(n: usize) -> Vec<Fr> {
    let mut v = Vec::with_capacity(n);
    let mut acc = Fr::from_u64(0x9E3779B97F4A7C15);
    let bump = Fr::from_u64(0xBF58476D1CE4E5B9);
    for _ in 0..n {
        acc = acc.mul(&bump).add(&bump);
        v.push(acc);
    }
    v
}

#[test]
#[ignore]
fn field_multiply_cost() {
    const N: usize = 20000;
    let xs = spread(N);
    let ys = spread(N);

    let wide = xs.iter().filter(|f| f.bit_len() >= 200).count();
    println!("operands with >= 200 bits: {} / {}", wide, N);
    assert!(wide > N / 2, "benchmark operands are not full-width field elements");

    let mut sink = Fr::one();
    for i in 0..N {
        sink = sink.add(&xs[i].mul(&ys[i]));
    }

    let rounds = 5;
    let mut best_mul = f64::MAX;
    for _ in 0..rounds {
        let t0 = Instant::now();
        for i in 0..N {
            let p = xs[i].mul(&ys[i]);
            std::hint::black_box(&p);
        }
        let per = t0.elapsed().as_secs_f64() / N as f64;
        if per < best_mul {
            best_mul = per;
        }
    }

    let mut best_add = f64::MAX;
    for _ in 0..rounds {
        let t0 = Instant::now();
        for i in 0..N {
            let p = xs[i].add(&ys[i]);
            std::hint::black_box(&p);
        }
        let per = t0.elapsed().as_secs_f64() / N as f64;
        if per < best_add {
            best_add = per;
        }
    }

    std::hint::black_box(&sink);
    println!("Fr::mul  {:>8.1} ns   (was 256-300 with the BigUint representation)", best_mul * 1e9);
    println!("Fr::add  {:>8.1} ns   (was 76-86)", best_add * 1e9);

    // What a Poseidon-chain emit spends in the field, using the counts the
    // compiler reports for 1000 hashes / 241,000 constraints.
    let muls = 4_122_000.0;
    let adds = 7_809_967.0;
    let field_s = muls * best_mul + adds * best_add;
    println!("\nfield time for 1000 Poseidon hashes: {:.3} s", field_s);
    println!("emit_program for the same circuit is ~1.9 s, of which ~1.2 s is optimize_circuit");
}

/// The field primitives must not allocate.
///
/// This is an assertion, not a measurement. The whole fix was to get `Fr` off
/// the heap: a single `Vec` reintroduced into `mul` or `add` puts back a
/// per-operation malloc/free pair on a path that runs 12 M times per 1000
/// hashes, and nothing else in the test suite would notice — the answers would
/// still be right, just an order of magnitude slower to produce.
#[test]
#[ignore]
fn field_primitives_do_not_allocate() {
    const N: usize = 20000;
    let xs = spread(N);
    let ys = spread(N);

    let a0 = ca::n();
    for i in 0..N {
        std::hint::black_box(xs[i].mul(&ys[i]));
    }
    let per_mul = (ca::n() - a0) as f64 / N as f64;

    let a0 = ca::n();
    for i in 0..N {
        std::hint::black_box(xs[i].add(&ys[i]));
    }
    let per_add = (ca::n() - a0) as f64 / N as f64;

    let a0 = ca::n();
    for i in 0..N {
        std::hint::black_box(xs[i]);
    }
    let per_copy = (ca::n() - a0) as f64 / N as f64;

    // A linear combination of the width a Poseidon constraint actually reaches
    // (~28 terms, from the 986 MB .r1cs for 964 k constraints), scaled and
    // folded the way the emitter does it. This one legitimately allocates: the
    // term vector and `simplify`'s map are real containers. It is the ceiling on
    // what is left to win above the field.
    let width = 28;
    let mut lc = LinearCombination::zero();
    for w in 0..width {
        lc.add_term(w + 1, xs[w]);
    }
    let a0 = ca::n();
    for i in 0..1000 {
        let mut t = lc.scale(ys[i % N]);
        t.add_linear(&lc, xs[i % N]);
        t.simplify();
        std::hint::black_box(&t);
    }
    let per_lc = (ca::n() - a0) as f64 / 1000.0;

    println!("allocations per Fr::mul                {:>8.1}  (was 22.2)", per_mul);
    println!("allocations per Fr::add                {:>8.1}  (was  6.4)", per_add);
    println!("allocations per Fr copy                {:>8.1}  (was  1.4)", per_copy);
    println!(
        "allocations per scale+add_linear+simplify (width {}) {:>6.1}  (was 1643)",
        width, per_lc
    );

    assert_eq!(per_mul, 0.0, "Fr::mul allocates; the representation has gone back on the heap");
    assert_eq!(per_add, 0.0, "Fr::add allocates; the representation has gone back on the heap");
    assert_eq!(per_copy, 0.0, "Fr is no longer Copy-with-no-drop-glue");
}
