//! M4, step 1: attention whose reduction is **order-independent by
//! construction**, demonstrated on the device across every launch geometry.
//!
//! `docs/deterministic_inference.md` M3 showed batch invariance for a matmul,
//! where exact integer accumulation makes the claim almost free. Attention is
//! the hard case, and it is the case the market actually complains about: the
//! standard FlashAttention/FlashInfer decode kernel carries an **online**
//! softmax whose running `(m, l, acc)` state is rescaled by
//! `exp(m_old - m_new)` once per tile, in floating point. Change the tiling —
//! which every server does, because `splits` and `warps` are tuned per batch
//! size — and you change the rescale sequence, and the answer moves in the low
//! bits. That is the mechanism behind "why did my eval score shift when I
//! reran it at a different batch size".
//!
//! ## The design being validated
//!
//! Two passes, no online state, and every reduction over the sequence carried
//! in exact integers:
//!
//! | step | operation | why it is order-independent |
//! |---|---|---|
//! | 1 | `s_i = q · k_i` | int8 x int8 -> int32, exact |
//! | 2 | `m = max_i s_i` | integer max: associative *and* commutative, exactly |
//! | 3 | `p_i = round(2^28 * 2^(C (s_i - m)))` | **per element**; no reduction |
//! | 4 | `l = sum_i p_i` | integer sum, exact |
//! | 5 | `o_d = sum_i p_i * v_id` | integer sum, exact |
//! | 6 | `out_d = o_d / l` | one float divide, at the very end |
//!
//! Steps 4 and 5 run as `red.global.add.u64` — a fire-and-forget atomic, the
//! single most notorious source of GPU nondeterminism — and are *still*
//! bit-identical, because integer addition cannot care what order it happened
//! in. Nothing is serialised and no determinism flag is set.
//!
//! Step 2 is what removes the online rescale: with the global max known before
//! any accumulation starts, `p_i` never has to be corrected. Costing a
//! separate pass is the honest price, and it is cheap in the shape that
//! matters — pass 2 re-reads `s` (4 bytes per key) rather than `K`
//! (`head_dim` bytes per key).
//!
//! ## What this does NOT claim
//!
//! `ex2.approx.f32` in step 3 is a fixed-function unit: same input, same
//! output, every launch, on a given architecture. That is sufficient for every
//! claim made here, because step 3 is per-element and so cannot depend on the
//! batch size or the partition. It is **not** sufficient for the cross-GPU
//! claim in M5, since a different SM generation may round it differently.
//! Closing that needs an integer exp (table plus fixed-point polynomial);
//! see the M4 notes in `docs/deterministic_inference.md`.
//!
//! Requires a CUDA driver; skipped with a notice otherwise.
//!
//! Run with: cargo test --release --test gpu_attention_invariance -- --nocapture

use y::cuda_runtime::CudaContext;

const D: usize = 32; // head dim
const S: usize = 256; // sequence length
const B: usize = 4; // query rows ("batch")

/// Softmax temperature, folded with log2(e) so `ex2` can be used directly.
/// Exactly representable (2^-13), so the constant itself introduces no
/// rounding. The value is *derived*, not guessed: the score spread over this
/// data is about -92,000, and 2^-13 puts the tail at 2^-11.3 — small enough to
/// be a real softmax, large enough that the smallest weight does not quantise
/// to zero and take a term out of the reduction entirely. The first attempt
/// used 2^-8 on an estimated spread of -1,500 and collapsed the distribution
/// to 15 distinct values; the degeneracy assertion below is what caught it.
const C: f32 = 0.000_122_070_312_5;
// `C_HEX` lived here, the same temperature as an f32 hex literal for the
// kernel template. The kernel takes the temperature as a runtime parameter
// now, so `$C` appears nowhere in the template and `attention_ptx` no longer
// accepts it -- the argument had been threaded through a `.replace` that
// matched nothing. `C` above is still used, by the host reference below.
/// Fixed-point scale for `p`, **derived in `attention_quantization_error.rs`**.
/// 2^16 was the first choice and was catastrophic on an attention sink: every
/// non-sink weight rounded to zero and the tail vanished, 2109x worse than f32
/// flash attention. At 2^28 the exact path is 0.01-0.20x flash's error on every
/// shape tested. Range obligation: p <= 2^28 fits an i32, `p*v` is 35 bits, and
/// the i64 accumulator holds sequences to 2^28 keys.
const P_HEX: &str = "0f4D800000"; // 2^28

fn ptx() -> String {
    attention_ptx(D, S).expect("D and S are within the exactness bound")
}

/// The module header plus the integer `exp2`, taken straight from
/// `src/fixed_exp.rs` so the kernel and the host reference cannot drift.
use y::exact_attention::attention_ptx;


fn inputs() -> (Vec<i8>, Vec<i8>, Vec<i8>) {
    let q = (0..B * D)
        .map(|i| (((i * 37 + 11) % 251) as i32 - 125) as i8)
        .collect();
    let k = (0..S * D)
        .map(|i| (((i * 53 + 7) % 251) as i32 - 125) as i8)
        .collect();
    let v = (0..S * D)
        .map(|i| (((i * 97 + 29) % 251) as i32 - 125) as i8)
        .collect();
    (q, k, v)
}

struct Run {
    p: Vec<i32>,
    l: Vec<i64>,
    o: Vec<i64>,
}

#[test]
fn attention_is_bit_identical_across_every_launch_geometry() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — attention invariance was not demonstrated.");
        return;
    };
    let src = ptx();
    let scores_mod = ctx.load_ptx(&src, "attn_scores").expect("attn_scores");
    let accum_mod = ctx.load_ptx(&src, "attn_accum").expect("attn_accum");
    let naive_mod = ctx
        .load_ptx(&src, "attn_accum_naive")
        .expect("attn_accum_naive");

    let (q, k, v) = inputs();
    let as_u8 = |x: &[i8]| x.iter().map(|&b| b as u8).collect::<Vec<u8>>();

    let d_q = ctx.alloc(B * D).unwrap();
    let d_k = ctx.alloc(S * D).unwrap();
    let d_v = ctx.alloc(S * D).unwrap();
    let d_s = ctx.alloc(B * S * 4).unwrap();
    let d_m = ctx.alloc(B * 4).unwrap();
    let d_l = ctx.alloc(B * 8).unwrap();
    let d_o = ctx.alloc(B * D * 8).unwrap();
    let d_p = ctx.alloc(B * S * 4).unwrap();
    ctx.memcpy_htod_at(&d_q, 0, &as_u8(&q)).unwrap();
    ctx.memcpy_htod_at(&d_k, 0, &as_u8(&k)).unwrap();
    ctx.memcpy_htod_at(&d_v, 0, &as_u8(&v)).unwrap();

    let run = |block: u32, gx: u32, gz: u32, naive: bool| -> Run {
        // Every accumulator this module writes is seeded the same way, M
        // included: `attn_scores` biases its max into unsigned space, whose
        // identity is 0. There is no i32::MIN pattern to remember.
        ctx.memset_u8(&d_m, 0).unwrap();
        ctx.memset_u8(&d_l, 0).unwrap();
        ctx.memset_u8(&d_o, 0).unwrap();
        ctx.memset_u8(&d_p, 0).unwrap();

        let sargs = vec![
            d_q.device_ptr(),
            d_k.device_ptr(),
            d_s.device_ptr(),
            d_m.device_ptr(),
        ];
        ctx.launch(
            &scores_mod,
            ((S as u32).div_ceil(128), B as u32, 1),
            (128, 1, 1),
            0,
            &sargs,
        )
        .expect("attn_scores launch");
        ctx.synchronize().unwrap();

        let aargs = vec![
            d_s.device_ptr(),
            d_v.device_ptr(),
            d_m.device_ptr(),
            d_l.device_ptr(),
            d_o.device_ptr(),
            d_p.device_ptr(),
            // C = 2^-13, so the old hardcoded `shl 3` is exactly 8 in Q16.16.
            // Passing it keeps this test's semantics identical while the
            // kernel becomes usable on a real model's arbitrary scale.
            8u64 << 16,
        ];
        let which = if naive { &naive_mod } else { &accum_mod };
        ctx.launch(which, (gx, B as u32, gz), (block, 1, 1), 0, &aargs)
            .expect("attn_accum launch");
        ctx.synchronize().unwrap();

        let mut raw_p = vec![0u8; B * S * 4];
        let mut raw_l = vec![0u8; B * 8];
        let mut raw_o = vec![0u8; B * D * 8];
        ctx.memcpy_dtoh_at(&mut raw_p, &d_p, 0).unwrap();
        ctx.memcpy_dtoh_at(&mut raw_l, &d_l, 0).unwrap();
        ctx.memcpy_dtoh_at(&mut raw_o, &d_o, 0).unwrap();
        let i32s = |r: &[u8]| {
            r.chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<i32>>()
        };
        let i64s = |r: &[u8]| {
            r.chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<i64>>()
        };
        Run {
            p: i32s(&raw_p),
            l: i64s(&raw_l),
            o: i64s(&raw_o),
        }
    };

    let reference = run(128, (S as u32).div_ceil(128), 1, false);

    // The oracle. `p` is a per-element function of `s`, so it is not what is in
    // question here — but `l` and `o` are reductions over it, and recomputing
    // them serially on the host from the device's own `p` pins the reductions
    // EXACTLY. Without this, "every geometry agrees" could hold with all of
    // them agreeing on a wrong answer.
    for b in 0..B {
        let want_l: i64 = (0..S).map(|i| reference.p[b * S + i] as i64).sum();
        assert_eq!(
            reference.l[b], want_l,
            "l[{b}] disagrees with a serial host sum over the device's own p"
        );
        for d in 0..D {
            let want_o: i64 = (0..S)
                .map(|i| reference.p[b * S + i] as i64 * v[i * D + d] as i64)
                .sum();
            assert_eq!(
                reference.o[b * D + d],
                want_o,
                "o[{b}][{d}] disagrees with a serial host sum over the device's own p"
            );
        }
    }

    // The weights must actually be spread out. A softmax that has collapsed to
    // one-hot, or flattened to uniform, would make every reduction below
    // trivial — and the temperature is a hand-picked constant, so this is a
    // real risk rather than a ceremonial check.
    let row: Vec<i32> = reference.p[0..S].to_vec();
    let hi = *row.iter().max().unwrap();
    let lo = *row.iter().min().unwrap();
    let distinct = {
        let mut s = row.clone();
        s.sort_unstable();
        s.dedup();
        s.len()
    };
    assert_eq!(
        hi,
        1 << 28,
        "the max-score weight should quantise to exactly 2^28"
    );
    assert!(
        lo * 16 < hi && distinct > S / 2,
        "the softmax is degenerate (min {lo}, {distinct} distinct of {S}); the \
         reductions below would then be measuring nothing"
    );

    // blockDim.x, gridDim.x and gridDim.z each change which thread accumulates
    // which term, and each is swept. Repeats matter because atomic contention
    // order is not reproducible run to run.
    let geometries: &[(u32, u32, u32)] = &[
        (128, 2, 1),
        (32, 8, 1),
        (32, 1, 8),
        (64, 3, 5),
        (256, 1, 1),
        (256, 4, 3),
        (32, 1, 1),
        (128, 1, 32),
        (64, 7, 2),
    ];
    for &(block, gx, gz) in geometries {
        for rep in 0..3 {
            // Both reduction STRUCTURES, at every geometry. The two-level
            // kernel groups terms per thread, then per CTA, then globally; the
            // naive one sends every term to a global atomic on its own. They
            // must agree with each other and with the reference, which is a
            // stronger statement than either being self-consistent.
            for naive in [false, true] {
                let got = run(block, gx, gz, naive);
                let tag = if naive { "naive" } else { "two-level" };
                assert_eq!(
                    got.p, reference.p,
                    "{tag} block {block} grid ({gx},{gz}) rep {rep}: the per-element weights moved"
                );
                assert_eq!(
                    got.l, reference.l,
                    "{tag} block {block} grid ({gx},{gz}) rep {rep}: the softmax denominator \
                     moved. Exact integer accumulation is supposed to make the partition and the \
                     reduction shape invisible."
                );
                assert_eq!(
                    got.o, reference.o,
                    "{tag} block {block} grid ({gx},{gz}) rep {rep}: the attention numerator \
                     moved. This is the product claim failing on the kernel that matters most."
                );
            }
        }
        let workers = block * gx * gz;
        eprintln!(
            "block {block:3}  grid.x {gx:2}  grid.z {gz:2}  ({workers:5} workers): \
             two-level == naive == reference, x3"
        );
    }

    // And it is attention, not merely something reproducible. Compared against
    // a plain f64 softmax over the same scores, with the same temperature.
    let mut worst = 0.0f64;
    for b in 0..B {
        let s: Vec<i32> = (0..S)
            .map(|i| (0..D).map(|d| q[b * D + d] as i32 * k[i * D + d] as i32).sum())
            .collect();
        let m = *s.iter().max().unwrap();
        let w: Vec<f64> = s
            .iter()
            .map(|&x| ((x - m) as f64 * C as f64).exp2())
            .collect();
        let wsum: f64 = w.iter().sum();
        for d in 0..D {
            let want: f64 =
                w.iter().enumerate().map(|(i, &x)| x * v[i * D + d] as f64).sum::<f64>() / wsum;
            let got = reference.o[b * D + d] as f64 / reference.l[b] as f64;
            worst = worst.max((want - got).abs());
        }
    }
    assert!(
        worst < 0.1,
        "the exact kernel is reproducible but is not computing attention: worst \
         element differs from an f64 softmax by {worst}"
    );
    eprintln!("agrees with an f64 softmax to {worst:.2e} absolute (values are in +-127)");
}

/// The two-level reduction's barriers are pinned STRUCTURALLY, because the
/// device cannot pin one of them.
///
/// Mutation testing found that deleting the barrier between zeroing the shared
/// accumulators and using them leaves the whole file green — five runs out of
/// five. The race is real (a thread may add into a slot another thread has not
/// zeroed yet) but its window never opens: between the zero loop and the first
/// `red.shared.add` sit a global load of `Scores`, an `ex2` chain and a global
/// store of `P`, so every warp is long done zeroing before any warp reaches an
/// accumulate. This is the same shape as the `bar.sync` finding recorded in
/// `CLAUDE.md` gotcha #8 — **a race that does not fire is not a test** — and
/// the same answer applies: assert the ordering in the emitted PTX, since the
/// guarantee is absent rather than merely unused, and a scheduling change or a
/// larger `D` would open the window.
///
/// The other barrier (before the global flush) IS caught on the device, 0/5.
#[test]
fn the_two_level_reduction_is_ordered_by_real_barriers() {
    let src = ptx();
    let start = src
        .find(".visible .entry attn_accum(")
        .expect("attn_accum entry");
    let body = &src[start..];
    let end = body[1..].find(".visible .entry").map(|i| i + 1).unwrap();
    let body = &body[..end];

    let at = |needle: &str| body.find(needle).unwrap_or_else(|| panic!("no {needle}"));
    let zero_done = at("ZSKIP:");
    let accumulate = at("LOOP_I:");
    let flush = at("FLOOP:");
    assert!(
        zero_done < accumulate && accumulate < flush,
        "the kernel's phases are not in the order this test assumes"
    );

    let bars: Vec<usize> = body
        .match_indices("bar.sync")
        .map(|(i, _)| i)
        .collect();
    assert!(
        bars.iter().any(|&b| b > zero_done && b < accumulate),
        "no barrier between zeroing the shared accumulators and accumulating into \
         them. The device test does NOT catch this — the race window never opens \
         at this size — so removing it here removes the only guard there is."
    );
    assert!(
        bars.iter().any(|&b| b > accumulate && b < flush),
        "no barrier between the last shared accumulate and the global flush"
    );
}

/// The control: the standard online-softmax attention, in f32, is NOT
/// invariant to the same partition changes.
///
/// This mirrors what a FlashAttention-style split-K decode kernel does — each
/// split keeps a running `(m, l, acc)` and the splits are combined by
/// rescaling to a common max — and shows that changing only the split count
/// changes the answer. Without this, the test above would be a statement about
/// benign data rather than about exact accumulation.
#[test]
fn the_online_softmax_in_f32_is_not_partition_invariant() {
    let (q, k, v) = inputs();

    // One split's running state, exactly as a flash kernel carries it.
    let split = |b: usize, z: usize, nz: usize| -> (f32, f32, Vec<f32>) {
        let (mut m, mut l) = (f32::NEG_INFINITY, 0.0f32);
        let mut acc = vec![0.0f32; D];
        let mut i = z;
        while i < S {
            let s: i32 = (0..D)
                .map(|d| q[b * D + d] as i32 * k[i * D + d] as i32)
                .sum();
            let x = s as f32 * C;
            let mn = m.max(x);
            let corr = (m - mn).exp2();
            let p = (x - mn).exp2();
            l = l * corr + p;
            for d in 0..D {
                acc[d] = acc[d] * corr + p * v[i * D + d] as f32;
            }
            m = mn;
            i += nz;
        }
        (m, l, acc)
    };
    let combine = |b: usize, nz: usize| -> Vec<f32> {
        let parts: Vec<_> = (0..nz).map(|z| split(b, z, nz)).collect();
        let gm = parts.iter().fold(f32::NEG_INFINITY, |a, p| a.max(p.0));
        let (mut l, mut acc) = (0.0f32, vec![0.0f32; D]);
        for (pm, pl, pa) in &parts {
            let c = (pm - gm).exp2();
            l += pl * c;
            for d in 0..D {
                acc[d] += pa[d] * c;
            }
        }
        acc.iter().map(|a| a / l).collect()
    };

    let mut moved = 0usize;
    let mut total = 0usize;
    let base: Vec<Vec<f32>> = (0..B).map(|b| combine(b, 1)).collect();
    for &nz in &[2usize, 3, 5, 8, 16, 32] {
        for b in 0..B {
            let got = combine(b, nz);
            for d in 0..D {
                total += 1;
                if got[d].to_bits() != base[b][d].to_bits() {
                    moved += 1;
                }
            }
        }
    }
    assert!(
        moved > 0,
        "the f32 online-softmax control did not move under ANY split count, so \
         the invariance result above proves nothing about this data"
    );
    eprintln!("control: f32 online softmax moves on {moved} of {total} (element, split) pairs");
}


/// **No floating-point instruction may appear in the attention kernels.**
///
/// Determinism does not actually require avoiding floats — IEEE `div.rn` is
/// correctly rounded and fully specified, so it is bit-identical everywhere.
/// The hazards are narrower than "floats": reassociation in a reduction, and
/// operations the ISA declines to pin down. `ex2.approx.f32` is the second
/// kind — "approx" means the result is a property of the SM generation, which
/// is exactly how it broke the cross-GPU claim.
///
/// So this file could have kept a float or two and still been correct. It
/// keeps none, because the weaker rule is the one that is checkable: "is any
/// float here of a specified-and-non-reassociating kind" needs a judgement per
/// instruction, and "are there any floats" is a grep. The strictness is for
/// the reviewer's benefit, not the hardware's.
///
/// `src/fixed_exp.rs` is what makes this affordable — and it is not a
/// concession, it is 8.8x MORE accurate than the hardware unit it replaced.
#[test]
fn the_attention_kernels_contain_no_floating_point() {
    let src = ptx();
    let start = src.find(".visible .entry attn_scores").expect("attn_scores");
    let kernels = &src[start..];

    let banned = [
        ".f32", ".f64", "ex2", "lg2", "rcp.", "sqrt.", "div.rn.f", "fma.rn.f", "mul.f", "add.f",
    ];
    let mut found = Vec::new();
    for (i, line) in kernels.lines().enumerate() {
        let code = line.split("//").next().unwrap_or("");
        for b in banned {
            if code.contains(b) {
                found.push(format!("  line {}: {}", i + 1, line.trim()));
            }
        }
    }
    assert!(
        found.is_empty(),
        "floating point crept back into the attention kernels:\n{}\n\nThe softmax \
         path is integer end to end (src/fixed_exp.rs). If a float genuinely \
         belongs here, it must be a specified, non-reassociating operation AND \
         this test must be updated deliberately — not because it went red.",
        found.join("\n")
    );

    // Control: the banned list must be capable of matching, or this test would
    // pass on an empty string.
    assert!(
        src.contains("red.global") && kernels.contains("call.uni"),
        "the kernels no longer look like the ones this test is about"
    );
}

/// **`attn_scores` is launch-invariant too — and it was not.**
///
/// The test above sweeps the geometry of `attn_accum` and launches
/// `attn_scores` at one fixed, correct geometry (`ceil(S/128) x 128`). So the
/// scores kernel's own launch contract was never exercised, and it turned out
/// to be the opposite of the accumulate's: one thread per key with a bounds
/// guard and no loop, silently requiring `gridDim.x * blockDim.x >= S`.
///
/// Measured on an RTX 4070 Ti SUPER at S=512 before the fix: launched with 128
/// threads it left **768 of 1024 score slots stale and computed the wrong
/// maximum** (34647 against 34752). The stale scores are the lesser half — the
/// maximum is subtracted from every score in `attn_accum`, so a wrong one moves
/// **every** softmax weight, and the failure is silent.
///
/// `attn_scores` is a grid-stride loop now, the same residue-class partition
/// `proofs/GridStrideSplit.v` proves. This test is what keeps it one: the
/// deliberately-short geometries below are the ones that used to be wrong.
#[test]
fn the_scores_kernel_is_launch_invariant_too() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — scores launch invariance was not demonstrated.");
        return;
    };
    let src = ptx();
    let scores_mod = ctx.load_ptx(&src, "attn_scores").expect("attn_scores");
    let (q, k, _) = inputs();
    let as_u8 = |x: &[i8]| x.iter().map(|&b| b as u8).collect::<Vec<u8>>();

    let d_q = ctx.alloc(B * D).unwrap();
    let d_k = ctx.alloc(S * D).unwrap();
    let d_s = ctx.alloc(B * S * 4).unwrap();
    let d_m = ctx.alloc(B * 4).unwrap();
    ctx.memcpy_htod_at(&d_q, 0, &as_u8(&q)).unwrap();
    ctx.memcpy_htod_at(&d_k, 0, &as_u8(&k)).unwrap();

    let run = |gx: u32, block: u32| -> (Vec<i32>, Vec<i32>) {
        ctx.memset_u8(&d_m, 0).unwrap();
        // NOT zeroed: a kernel that skips a key must show as a stale slot, and
        // zeroing here would hide exactly the failure this test exists for.
        ctx.memset_u8(&d_s, 0xAB).unwrap();
        let args =
            vec![d_q.device_ptr(), d_k.device_ptr(), d_s.device_ptr(), d_m.device_ptr()];
        ctx.launch(&scores_mod, (gx, B as u32, 1), (block, 1, 1), 0, &args)
            .expect("attn_scores launch");
        ctx.synchronize().unwrap();
        let mut rs = vec![0u8; B * S * 4];
        let mut rm = vec![0u8; B * 4];
        ctx.memcpy_dtoh_at(&mut rs, &d_s, 0).unwrap();
        ctx.memcpy_dtoh_at(&mut rm, &d_m, 0).unwrap();
        let to_i32 =
            |v: &[u8]| v.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        (to_i32(&rs), to_i32(&rm))
    };

    // The reference is a geometry that covers the sequence, which is what the
    // kernel used to require.
    let (s_ref, m_ref) = run((S as u32).div_ceil(128), 128);
    assert!(m_ref.iter().all(|&m| m != 0), "the reference launch computed no maximum");

    // Every one of these is SHORT of S, so each thread must take several keys.
    let geometries: &[(u32, u32)] = &[(1, 32), (1, 64), (1, 128), (2, 32), (3, 64), (7, 16)];
    for &(gx, block) in geometries {
        let covered = gx as usize * block as usize;
        assert!(covered < S, "({gx},{block}) covers {covered} >= S={S}; not a short launch");
        let (s, m) = run(gx, block);
        assert_eq!(
            s, s_ref,
            "attn_scores at grid.x {gx} block {block} ({covered} threads for S={S}) \
             does not agree with a covering launch. It is one thread per key \
             again, so keys past the launch are silently skipped"
        );
        assert_eq!(
            m, m_ref,
            "attn_scores at grid.x {gx} block {block} computed a different \
             maximum. That is the worse half: attn_accum subtracts it from every \
             score, so every softmax weight moves"
        );
    }
}

/// **The max reduction's IDENTITY, which was the last precondition this module
/// left to the host.**
///
/// `red.global.max` is associative, commutative and idempotent, so
/// `GridStrideSplit.v` covers the order and the partition. It says nothing
/// about the value the buffer starts at, and a signed max wants `i32::MIN` —
/// not a uniform byte pattern, so M could not be seeded by the same
/// `memset(0)` that L, O and P take. A caller who used one anyway got a
/// silently wrong answer **only when a row's scores were all negative**: with
/// `max(0, s) = 0` the softmax then subtracts a maximum no key attains, every
/// weight is scaled down by `2^(C*|shift|)`, and in Q0.28 they quantise toward
/// zero rather than cancelling as they would in exact arithmetic.
///
/// That is the worst shape a precondition can have — it cannot fire on random
/// int8 data, where the max over hundreds of keys is positive with
/// overwhelming probability, so every test in this file passed either way.
///
/// The kernel biases into unsigned space (`x ^ 0x80000000`, the
/// order-preserving signed→unsigned bijection), and unsigned max has identity
/// **0**. A zeroed buffer now *means* `i32::MIN`.
///
/// This test seeds M with a plain `memset(0)` over an input whose scores are
/// **all negative**, and asserts the kernel finds the true (negative) maximum.
/// Reverting the bias to `red.global.max.s32` leaves M at 0 and fails it.
#[test]
fn a_zeroed_max_buffer_is_the_identity() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — the max identity was not demonstrated.");
        return;
    };
    let src = ptx();
    let scores_mod = ctx.load_ptx(&src, "attn_scores").expect("attn_scores");

    // Every score is a dot product of a strictly positive Q row with a
    // strictly negative K row, so every score is strictly negative and the
    // identity is the only thing that can supply the maximum.
    let q: Vec<i8> = (0..B * D).map(|i| ((i % 100) + 1) as i8).collect();
    let k: Vec<i8> = (0..S * D).map(|i| -(((i % 100) + 1) as i8)).collect();

    let as_u8 = |x: &[i8]| x.iter().map(|&b| b as u8).collect::<Vec<u8>>();
    let d_q = ctx.alloc(B * D).unwrap();
    let d_k = ctx.alloc(S * D).unwrap();
    let d_s = ctx.alloc(B * S * 4).unwrap();
    let d_m = ctx.alloc(B * 4).unwrap();
    ctx.memcpy_htod_at(&d_q, 0, &as_u8(&q)).unwrap();
    ctx.memcpy_htod_at(&d_k, 0, &as_u8(&k)).unwrap();
    ctx.memset_u8(&d_s, 0xAB).unwrap();
    // The whole point: the same seed L, O and P get. No i32::MIN pattern.
    ctx.memset_u8(&d_m, 0).unwrap();

    let args = vec![d_q.device_ptr(), d_k.device_ptr(), d_s.device_ptr(), d_m.device_ptr()];
    ctx.launch(&scores_mod, ((S as u32).div_ceil(128), B as u32, 1), (128, 1, 1), 0, &args)
        .expect("attn_scores launch");
    ctx.synchronize().unwrap();

    let mut rs = vec![0u8; B * S * 4];
    let mut rm = vec![0u8; B * 4];
    ctx.memcpy_dtoh_at(&mut rs, &d_s, 0).unwrap();
    ctx.memcpy_dtoh_at(&mut rm, &d_m, 0).unwrap();
    let to_i32 = |v: &[u8]| -> Vec<i32> {
        v.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    let scores = to_i32(&rs);
    // M is stored biased; undo it exactly as the accumulating entries do.
    let maxima: Vec<i32> =
        to_i32(&rm).iter().map(|&m| ((m as u32) ^ 0x8000_0000) as i32).collect();

    for b in 0..B {
        let row = &scores[b * S..(b + 1) * S];
        let want = *row.iter().max().unwrap();

        // Non-vacuity: this case only bites when the true maximum is negative,
        // so a fixture whose scores drifted positive would pass while testing
        // nothing.
        assert!(
            want < 0,
            "row {b}'s maximum is {want}; the fixture is supposed to make every \
             score negative, and with a positive maximum `max(0, s)` is right \
             by accident and this test asserts nothing"
        );
        assert_eq!(
            maxima[b], want,
            "row {b}: the kernel found {} (raw {}) where the true maximum is \
             {want}. A zeroed M is not being read as the identity — a signed \
             max leaves the raw word at 0, which no key attains, and \
             `attn_accum` then subtracts that from every score",
            maxima[b],
            (maxima[b] as u32) ^ 0x8000_0000
        );
    }
}

/// **The other half of the bias, and the temperature is what makes it
/// testable.**
///
/// `attn_scores` stores M biased; `attn_accum` and `attn_accum_naive` must
/// undo it. Deleting the undo from ONE of the two entries passes every other
/// test in this file — including the accum-vs-naive differential on `p`, which
/// is precisely the comparison that ought to catch a one-sided change.
///
/// It survives for an arithmetic accident worth writing down. The bias is
/// `2^31`; the kernel computes `t = ((m - s) * KFix + 2^15) >> 16` in 64 bits
/// and then **truncates to u32**. A missing undo therefore shifts `t` by
/// `(2^31 * KFix) >> 16 = 2^15 * KFix`, and the other tests pass
/// `KFix = 8 << 16 = 2^19`, for which that shift is `2^34` — **exactly zero
/// modulo 2^32**. The truncation annihilates the bug. Every arm of the
/// differential shares the constant, so every arm is wrong in the same
/// invisible way.
///
/// A real model's softmax scale is `q_scale * k_scale / sqrt(d)`, an arbitrary
/// number. This test uses `2^19 + 1` — the same temperature to within 2e-6,
/// and odd, so the shift is `2^34 + 2^15 ≡ 2^15` and a missing undo drives
/// almost every weight to zero.
///
/// The oracle is `fixed_exp::exp2_neg_q16_16` on the host over the device's own
/// scores, which is what `p` was never checked against: the existing test
/// says `p` is "a per-element function of `s`, so it is not what is in
/// question" — true of the reduction claim, and the undo lives exactly there.
#[test]
fn the_accumulating_entries_undo_the_bias() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — the bias undo was not demonstrated.");
        return;
    };

    // Not a multiple of 2^17, which is the whole point: `2^15 * KFix` must not
    // vanish modulo 2^32, or a missing undo is invisible here too.
    const KFIX: u64 = (8u64 << 16) + 1;
    assert_ne!(
        (KFIX << 15) % (1u64 << 32),
        0,
        "the temperature annihilates the bias under the u32 truncation, so this \
         test cannot see a missing undo — pick one that is not a multiple of 2^17"
    );

    let src = ptx();
    let scores_mod = ctx.load_ptx(&src, "attn_scores").expect("attn_scores");
    let accum_mod = ctx.load_ptx(&src, "attn_accum").expect("attn_accum");
    let naive_mod = ctx.load_ptx(&src, "attn_accum_naive").expect("attn_accum_naive");

    let (q, k, v) = inputs();
    let as_u8 = |x: &[i8]| x.iter().map(|&b| b as u8).collect::<Vec<u8>>();
    let d_q = ctx.alloc(B * D).unwrap();
    let d_k = ctx.alloc(S * D).unwrap();
    let d_v = ctx.alloc(S * D).unwrap();
    let d_s = ctx.alloc(B * S * 4).unwrap();
    let d_m = ctx.alloc(B * 4).unwrap();
    let d_l = ctx.alloc(B * 8).unwrap();
    let d_o = ctx.alloc(B * D * 8).unwrap();
    let d_p = ctx.alloc(B * S * 4).unwrap();
    ctx.memcpy_htod_at(&d_q, 0, &as_u8(&q)).unwrap();
    ctx.memcpy_htod_at(&d_k, 0, &as_u8(&k)).unwrap();
    ctx.memcpy_htod_at(&d_v, 0, &as_u8(&v)).unwrap();
    ctx.memset_u8(&d_m, 0).unwrap();
    ctx.memset_u8(&d_s, 0).unwrap();

    let sargs = vec![d_q.device_ptr(), d_k.device_ptr(), d_s.device_ptr(), d_m.device_ptr()];
    ctx.launch(&scores_mod, ((S as u32).div_ceil(128), B as u32, 1), (128, 1, 1), 0, &sargs)
        .expect("attn_scores launch");
    ctx.synchronize().unwrap();

    let to_i32 = |v: &[u8]| -> Vec<i32> {
        v.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    let mut raw = vec![0u8; B * S * 4];
    ctx.memcpy_dtoh_at(&mut raw, &d_s, 0).unwrap();
    let scores = to_i32(&raw);
    let mut raw_m = vec![0u8; B * 4];
    ctx.memcpy_dtoh_at(&mut raw_m, &d_m, 0).unwrap();
    let maxima: Vec<i32> =
        to_i32(&raw_m).iter().map(|&m| ((m as u32) ^ 0x8000_0000) as i32).collect();

    let run_accum = |naive: bool| -> Vec<i32> {
        ctx.memset_u8(&d_l, 0).unwrap();
        ctx.memset_u8(&d_o, 0).unwrap();
        ctx.memset_u8(&d_p, 0).unwrap();
        let aargs = vec![
            d_s.device_ptr(),
            d_v.device_ptr(),
            d_m.device_ptr(),
            d_l.device_ptr(),
            d_o.device_ptr(),
            d_p.device_ptr(),
            KFIX,
        ];
        let which = if naive { &naive_mod } else { &accum_mod };
        ctx.launch(which, ((S as u32).div_ceil(128), B as u32, 1), (128, 1, 1), 0, &aargs)
            .expect("attn_accum launch");
        ctx.synchronize().unwrap();
        let mut r = vec![0u8; B * S * 4];
        ctx.memcpy_dtoh_at(&mut r, &d_p, 0).unwrap();
        to_i32(&r)
    };

    // The host reading of the kernel's own arithmetic, from `m` and `s`.
    let want = |b: usize, i: usize| -> i32 {
        let diff = maxima[b].wrapping_sub(scores[b * S + i]) as i64;
        let t = ((diff * KFIX as i64 + 32768) >> 16).min(1_073_741_824) as u32;
        y::fixed_exp::exp2_neg_q16_16(t) as i32
    };

    for (tag, p) in [("two-level", run_accum(false)), ("naive", run_accum(true))] {
        let mut nonzero = 0usize;
        for b in 0..B {
            for i in 0..S {
                let w = want(b, i);
                assert_eq!(
                    p[b * S + i],
                    w,
                    "{tag}: p[{b}][{i}] is {} where the host exp of the device's own \
                     score gives {w}. The most likely cause is a maximum that was \
                     never unbiased: `attn_scores` stores `m ^ 0x80000000`",
                    p[b * S + i]
                );
                if w != 0 {
                    nonzero += 1;
                }
            }
        }
        // Non-vacuity: `p == 0` everywhere agrees with a host oracle that is
        // also zero everywhere, so a collapsed softmax would make the loop
        // above assert nothing.
        assert!(
            nonzero > B * S / 2,
            "{tag}: only {nonzero} of {} weights are non-zero; the softmax has \
             collapsed and the comparison above is vacuous",
            B * S
        );
    }
}
