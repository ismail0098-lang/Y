//! M4 step 1c: **what does exactness cost in the decode regime?**
//!
//! `docs/deterministic_inference.md` §6 item 3 names this as a thing that
//! could kill the programme, and it has been asserted at "about 3%" from a
//! byte count without ever being run. This measures it.
//!
//! ## What is compared, and why it is a fair pair
//!
//! The attention accumulation phase, in the layout a decode kernel actually
//! uses: **one thread owns one output dimension `d` and walks the whole key
//! range**, so `V` is read perfectly coalesced (`D` consecutive bytes across
//! the block per key) and the `o` accumulator needs no cross-thread reduction
//! at all. Both variants use that identical layout, identical partition,
//! identical loads:
//!
//! | | exact (integer) | baseline (f32 online softmax) |
//! |---|---|---|
//! | per key | `acc += p_i * v` | `acc = acc*corr + p*v` |
//! | per key | — | `m' = max(m, x)`, two `ex2` |
//! | reads | `P` int32, `V` int8 | `S` int32, `V` int8 |
//! | reduction | integer, order-independent | float, rescale-ordered |
//!
//! The baseline is the genuine FlashAttention inner loop: a running
//! `(m, l, acc)` rescaled by `exp(m_old - m_new)`. Both read exactly the same
//! bytes, so this isolates the *arithmetic* cost of exactness.
//!
//! ## What is NOT measured here, stated plainly
//!
//! A production flash kernel computes the scores in the **same** pass, keeping
//! them in registers; the exact path needs them in a prior pass because the
//! global max must exist before any accumulation starts. That difference is
//! traffic, not arithmetic, and it is arithmetic to bound: both read `K` once,
//! so the exact path's *extra* traffic is one write plus one read of `s`,
//! `2*S*4` bytes against `2*S*D` for K and V, i.e. **`4/D`** — 3.1% at
//! `head_dim = 128`. This file does not measure a fused-score flash kernel and
//! therefore does not quote an end-to-end ratio against FlashInfer.
//!
//! Run with:
//!   cargo test --release --test gpu_attention_cost -- --ignored --nocapture

use y::cuda_runtime::CudaContext;

const D: usize = 128; // head dim
const S: usize = 4096; // sequence length
const B: usize = 32; // rows (batch x heads)

const C_HEX: &str = "0f39000000"; // 2^-13
const C: f32 = 0.000_122_070_312_5;

fn ptx() -> String {
    SRC.replace("$D", &D.to_string())
        .replace("$S", &S.to_string())
        .replace("$C", C_HEX)
}

const SRC: &str = r#"
.version 7.8
.target sm_89
.address_size 64

// Exact: thread d owns output dimension d, walks the key range, accumulates in
// a 64-bit integer register. No atomics in the loop, one per CTA at the end.
.visible .entry accum_exact(
    .param .u64 e0,   // P  int32 [B][S]
    .param .u64 e1,   // V  int8  [B][S][D]
    .param .u64 e2,   // L  u64   [B]
    .param .u64 e3    // O  s64   [B][D]
)
{
    .reg .pred %p<4>;
    .reg .s32  %r<32>;
    .reg .s64  %rd<32>;

    ld.param.u64 %rd1, [e0];
    ld.param.u64 %rd2, [e1];
    ld.param.u64 %rd3, [e2];
    ld.param.u64 %rd4, [e3];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    cvta.to.global.u64 %rd4, %rd4;

    mov.u32 %r1, %tid.x;                // d
    mov.u32 %r2, %ctaid.y;              // b
    mov.u32 %r3, %ctaid.z;              // i0
    mov.u32 %r4, %nctaid.z;             // istride

    mul.lo.s32 %r5, %r2, $S;
    mul.wide.s32 %rd5, %r5, 4;
    add.s64 %rd6, %rd1, %rd5;           // &P[b][0]

    mul.lo.s32 %r6, %r2, $S;
    mul.wide.s32 %rd7, %r6, $D;
    add.s64 %rd8, %rd2, %rd7;
    cvt.s64.s32 %rd9, %r1;
    add.s64 %rd8, %rd8, %rd9;           // &V[b][0][d]

    mov.u64 %rd10, 0;                   // acc
    mov.u64 %rd11, 0;                   // l
    mov.u32 %r7, %r3;                   // i
LOOP_E:
    setp.ge.s32 %p1, %r7, $S;
    @%p1 bra DONE_E;

    mul.wide.s32 %rd12, %r7, 4;
    add.s64 %rd13, %rd6, %rd12;
    ld.global.s32 %r8, [%rd13];         // p_i (broadcast across the block)

    mul.wide.s32 %rd14, %r7, $D;
    add.s64 %rd15, %rd8, %rd14;
    ld.global.s8 %r9, [%rd15];          // v[i][d] (coalesced across the block)

    mul.wide.s32 %rd16, %r8, %r9;       // 35 bits at F=28
    add.s64 %rd10, %rd10, %rd16;
    cvt.s64.s32 %rd17, %r8;
    add.s64 %rd11, %rd11, %rd17;

    add.s32 %r7, %r7, %r4;
    bra LOOP_E;
DONE_E:
    mul.lo.s32 %r11, %r2, $D;
    add.s32 %r11, %r11, %r1;
    mul.wide.s32 %rd18, %r11, 8;
    add.s64 %rd19, %rd4, %rd18;
    red.global.add.u64 [%rd19], %rd10;

    setp.ne.s32 %p2, %r1, 0;
    @%p2 bra SKIP_E;
    mul.wide.s32 %rd20, %r2, 8;
    add.s64 %rd21, %rd3, %rd20;
    red.global.add.u64 [%rd21], %rd11;
SKIP_E:
    ret;
}

// Baseline: the FlashAttention inner loop. Same layout, same loads, running
// (m, l, acc) rescaled by exp(m_old - m_new) in f32.
.visible .entry accum_f32(
    .param .u64 f0,   // S32 int32 [B][S]
    .param .u64 f1,   // V   int8  [B][S][D]
    .param .u64 f2,   // Lp  f32   [B][NZ]
    .param .u64 f3,   // Op  f32   [B][NZ][D]
    .param .u64 f4    // Mp  f32   [B][NZ]
)
{
    .reg .pred %p<4>;
    .reg .s32  %r<32>;
    .reg .s64  %rd<32>;
    .reg .f32  %f<20>;

    ld.param.u64 %rd1, [f0];
    ld.param.u64 %rd2, [f1];
    ld.param.u64 %rd3, [f2];
    ld.param.u64 %rd4, [f3];
    ld.param.u64 %rd5, [f4];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    cvta.to.global.u64 %rd4, %rd4;
    cvta.to.global.u64 %rd5, %rd5;

    mov.u32 %r1, %tid.x;                // d
    mov.u32 %r2, %ctaid.y;              // b
    mov.u32 %r3, %ctaid.z;              // z
    mov.u32 %r4, %nctaid.z;             // nz

    mul.lo.s32 %r5, %r2, $S;
    mul.wide.s32 %rd6, %r5, 4;
    add.s64 %rd7, %rd1, %rd6;           // &S32[b][0]

    mul.lo.s32 %r6, %r2, $S;
    mul.wide.s32 %rd8, %r6, $D;
    add.s64 %rd9, %rd2, %rd8;
    cvt.s64.s32 %rd10, %r1;
    add.s64 %rd9, %rd9, %rd10;          // &V[b][0][d]

    mov.f32 %f1, 0fFF800000;            // m = -inf
    mov.f32 %f2, 0f00000000;            // l
    mov.f32 %f3, 0f00000000;            // acc
    mov.u32 %r7, %r3;                   // i
LOOP_F:
    setp.ge.s32 %p1, %r7, $S;
    @%p1 bra DONE_F;

    mul.wide.s32 %rd11, %r7, 4;
    add.s64 %rd12, %rd7, %rd11;
    ld.global.s32 %r8, [%rd12];
    cvt.rn.f32.s32 %f4, %r8;
    mul.f32 %f5, %f4, $C;               // x

    max.f32 %f6, %f1, %f5;              // m' = max(m, x)
    sub.f32 %f7, %f1, %f6;
    ex2.approx.f32 %f8, %f7;            // corr = 2^(m - m')
    sub.f32 %f9, %f5, %f6;
    ex2.approx.f32 %f10, %f9;           // p = 2^(x - m')

    fma.rn.f32 %f2, %f2, %f8, %f10;     // l = l*corr + p

    mul.wide.s32 %rd13, %r7, $D;
    add.s64 %rd14, %rd9, %rd13;
    ld.global.s8 %r9, [%rd14];
    cvt.rn.f32.s32 %f11, %r9;
    mul.f32 %f12, %f10, %f11;
    fma.rn.f32 %f3, %f3, %f8, %f12;     // acc = acc*corr + p*v

    mov.f32 %f1, %f6;
    add.s32 %r7, %r7, %r4;
    bra LOOP_F;
DONE_F:
    // partial (m, l, acc) for this split, combined on the host
    mul.lo.s32 %r10, %r2, %r4;
    add.s32 %r10, %r10, %r3;            // b*NZ + z
    mul.wide.s32 %rd15, %r10, 4;
    add.s64 %rd16, %rd3, %rd15;
    add.s64 %rd17, %rd5, %rd15;

    mul.lo.s32 %r11, %r10, $D;
    add.s32 %r11, %r11, %r1;
    mul.wide.s32 %rd18, %r11, 4;
    add.s64 %rd19, %rd4, %rd18;
    st.global.f32 [%rd19], %f3;

    setp.ne.s32 %p2, %r1, 0;
    @%p2 bra SKIP_F;
    st.global.f32 [%rd16], %f2;
    st.global.f32 [%rd17], %f1;
SKIP_F:
    ret;
}
// The fair twin: identical structure to accum_exact, identical loads, but the
// accumulator is f32. No ex2 and no running max -- p is precomputed exactly as
// the exact path's is. The ONLY difference from accum_exact is int vs float.
.visible .entry accum_f32_twin(
    .param .u64 g0,   // Pf f32   [B][S]
    .param .u64 g1,   // V  int8  [B][S][D]
    .param .u64 g2,   // Lp f32   [B][NZ]
    .param .u64 g3    // Op f32   [B][NZ][D]
)
{
    .reg .pred %p<4>;
    .reg .s32  %r<32>;
    .reg .s64  %rd<32>;
    .reg .f32  %f<12>;

    ld.param.u64 %rd1, [g0];
    ld.param.u64 %rd2, [g1];
    ld.param.u64 %rd3, [g2];
    ld.param.u64 %rd4, [g3];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    cvta.to.global.u64 %rd4, %rd4;

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.y;
    mov.u32 %r3, %ctaid.z;
    mov.u32 %r4, %nctaid.z;

    mul.lo.s32 %r5, %r2, $S;
    mul.wide.s32 %rd5, %r5, 4;
    add.s64 %rd6, %rd1, %rd5;

    mul.lo.s32 %r6, %r2, $S;
    mul.wide.s32 %rd7, %r6, $D;
    add.s64 %rd8, %rd2, %rd7;
    cvt.s64.s32 %rd9, %r1;
    add.s64 %rd8, %rd8, %rd9;

    mov.f32 %f1, 0f00000000;            // acc
    mov.f32 %f2, 0f00000000;            // l
    mov.u32 %r7, %r3;
LOOP_T:
    setp.ge.s32 %p1, %r7, $S;
    @%p1 bra DONE_T;

    mul.wide.s32 %rd10, %r7, 4;
    add.s64 %rd11, %rd6, %rd10;
    ld.global.f32 %f3, [%rd11];

    mul.wide.s32 %rd12, %r7, $D;
    add.s64 %rd13, %rd8, %rd12;
    ld.global.s8 %r8, [%rd13];
    cvt.rn.f32.s32 %f4, %r8;

    fma.rn.f32 %f1, %f3, %f4, %f1;
    add.f32 %f2, %f2, %f3;

    add.s32 %r7, %r7, %r4;
    bra LOOP_T;
DONE_T:
    mul.lo.s32 %r10, %r2, %r4;
    add.s32 %r10, %r10, %r3;
    mul.wide.s32 %rd14, %r10, 4;
    add.s64 %rd15, %rd3, %rd14;
    mul.lo.s32 %r11, %r10, $D;
    add.s32 %r11, %r11, %r1;
    mul.wide.s32 %rd16, %r11, 4;
    add.s64 %rd17, %rd4, %rd16;
    st.global.f32 [%rd17], %f1;
    setp.ne.s32 %p2, %r1, 0;
    @%p2 bra SKIP_T;
    st.global.f32 [%rd15], %f2;
SKIP_T:
    ret;
}
"#;

/// Scores as a decode kernel would have produced them, the exact path's
/// quantised weights derived from the same scores, and V. Both variants are
/// therefore fed the same distribution.
fn data() -> (Vec<i32>, Vec<i32>, Vec<u8>) {
    let scores: Vec<i32> = (0..B * S)
        .map(|i| -(((i * 2_654_435_761usize) % 90_000) as i32))
        .collect();
    let mut weights = vec![0i32; B * S];
    for b in 0..B {
        let m = *scores[b * S..(b + 1) * S].iter().max().unwrap();
        for i in 0..S {
            let x = (scores[b * S + i] - m) as f32 * C;
            weights[b * S + i] = (x.exp2() * 268_435_456.0).round() as i32; // 2^28
        }
    }
    let v: Vec<u8> = (0..B * S * D)
        .map(|i| ((((i * 97 + 29) % 251) as i32 - 125) as i8) as u8)
        .collect();
    (scores, weights, v)
}

/// Bytes each variant reads in the accumulation phase. Identical by
/// construction: `V` once, and one int32 per key.
const ACCUM_BYTES: usize = B * S * D + B * S * 4;

#[test]
#[ignore = "benchmark; requires a CUDA device"]
fn what_exactness_costs_in_the_decode_accumulation() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let src = ptx();
    let exact = ctx.load_ptx(&src, "accum_exact").expect("accum_exact");
    let flash = ctx.load_ptx(&src, "accum_f32").expect("accum_f32");
    let twin = ctx.load_ptx(&src, "accum_f32_twin").expect("accum_f32_twin");

    let (scores, weights, v) = data();

    let d_p = ctx.alloc(B * S * 4).unwrap();
    let d_s = ctx.alloc(B * S * 4).unwrap();
    let d_v = ctx.alloc(B * S * D).unwrap();
    let d_l = ctx.alloc(B * 8).unwrap();
    let d_o = ctx.alloc(B * D * 8).unwrap();
    let bytes32 = |x: &[i32]| x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    ctx.memcpy_htod_at(&d_p, 0, &bytes32(&weights)).unwrap();
    ctx.memcpy_htod_at(&d_s, 0, &bytes32(&scores)).unwrap();
    ctx.memcpy_htod_at(&d_v, 0, &v).unwrap();

    let d_pf = ctx.alloc(B * S * 4).unwrap();
    let pf: Vec<u8> = weights
        .iter()
        .flat_map(|&w| (w as f32).to_le_bytes())
        .collect();
    ctx.memcpy_htod_at(&d_pf, 0, &pf).unwrap();

    let max_nz = 32usize;
    let d_lp = ctx.alloc(B * max_nz * 4).unwrap();
    let d_mp = ctx.alloc(B * max_nz * 4).unwrap();
    let d_op = ctx.alloc(B * max_nz * D * 4).unwrap();

    // `time_launches` returns MICROSECONDS per launch, not seconds.
    let gbps = |us: f64| ACCUM_BYTES as f64 / (us * 1e-6) / 1e9;

    // Peak DRAM bandwidth from the device itself, so "is this memory-bound"
    // is checkable rather than asserted.
    const MEMORY_CLOCK_RATE: i32 = 36; // kHz
    const GLOBAL_MEMORY_BUS_WIDTH: i32 = 37; // bits
    let peak = match (
        ctx.device_attribute(MEMORY_CLOCK_RATE),
        ctx.device_attribute(GLOBAL_MEMORY_BUS_WIDTH),
    ) {
        (Some(khz), Some(bits)) => khz as f64 * 1e3 * 2.0 * (bits as f64 / 8.0) / 1e9,
        _ => 0.0,
    };

    eprintln!(
        "decode accumulation, head_dim {D}, seq {S}, rows {B}  ({:.1} MB read per pass)",
        ACCUM_BYTES as f64 / 1e6
    );
    eprintln!("{} peak DRAM {peak:.0} GB/s", ctx.device_name());
    eprintln!(
        "{:>6} {:>11} {:>8} {:>7} {:>11} {:>8} {:>7}   ratio",
        "splits", "exact us", "GB/s", "%peak", "online us", "GB/s", "%peak"
    );
    eprintln!("   (`twin` = f32 with the SAME precomputed p: isolates int vs float alone)");

    let mut ratios = Vec::new();
    let mut best_exact = 0.0f64;
    let mut twin_ratios: Vec<f64> = Vec::new();
    for &nz in &[1u32, 4, 8, 16, 32] {
        let eargs = vec![vec![
            d_p.device_ptr(),
            d_v.device_ptr(),
            d_l.device_ptr(),
            d_o.device_ptr(),
        ]];
        let targs = vec![vec![
            d_pf.device_ptr(),
            d_v.device_ptr(),
            d_lp.device_ptr(),
            d_op.device_ptr(),
        ]];
        let fargs = vec![vec![
            d_s.device_ptr(),
            d_v.device_ptr(),
            d_lp.device_ptr(),
            d_op.device_ptr(),
            d_mp.device_ptr(),
        ]];
        // Warm the clocks before either is timed: this repo has a recorded
        // 38x error from a missing ramp, and an A/B is where it bites hardest.
        ctx.time_launches(&exact, (1, B as u32, nz), (D as u32, 1, 1), 0, &eargs, 50)
            .unwrap();
        ctx.time_launches(&flash, (1, B as u32, nz), (D as u32, 1, 1), 0, &fargs, 50)
            .unwrap();

        // Interleaved rounds, minimum taken -- ranking by the minimum is this
        // repo's rule, because the noise is one-sided.
        ctx.time_launches(&twin, (1, B as u32, nz), (D as u32, 1, 1), 0, &targs, 50)
            .unwrap();
        let (mut te, mut tf, mut tt) = (f64::MAX, f64::MAX, f64::MAX);
        for _ in 0..5 {
            tt = tt.min(
                ctx.time_launches(&twin, (1, B as u32, nz), (D as u32, 1, 1), 0, &targs, 100)
                    .unwrap(),
            );
            te = te.min(
                ctx.time_launches(&exact, (1, B as u32, nz), (D as u32, 1, 1), 0, &eargs, 100)
                    .unwrap(),
            );
            tf = tf.min(
                ctx.time_launches(&flash, (1, B as u32, nz), (D as u32, 1, 1), 0, &fargs, 100)
                    .unwrap(),
            );
        }
        eprintln!(
            "{nz:>6} {:>11.1} {:>8.1} {:>6.0}% {:>11.1} {:>8.1} {:>6.0}%                vs online {:.2}x   vs twin {:.2}x",
            te,
            gbps(te),
            100.0 * gbps(te) / peak,
            tf,
            gbps(tf),
            100.0 * gbps(tf) / peak,
            te / tf,
            te / tt
        );
        twin_ratios.push(te / tt);
        ratios.push(te / tf);
        best_exact = best_exact.max(gbps(te));
    }

    let tw = twin_ratios.iter().cloned().fold(0.0f64, f64::max);
    eprintln!();
    eprintln!(
        "AGAINST THE FAIR TWIN (same precomputed p, f32 accumulator): exact is at \
         worst {tw:.2}x. This is the number to quote for the arithmetic; the \
         `online` column charges f32 for ex2 work the exact path did in an \
         untimed earlier pass, and charges it 128x over because every thread in \
         the block recomputes the same per-key scalars."
    );
    let worst = ratios.iter().cloned().fold(0.0f64, f64::max);
    eprintln!();
    eprintln!(
        "(against the per-key online softmax it reads {worst:.2}x, but that baseline \
         is charged for ex2 the exact path paid earlier, 128x redundantly -- do not \
         quote it.)"
    );
    eprintln!(
        "exact peaks at {best_exact:.0} GB/s = {:.0}% of DRAM peak, so the phase is \
         bandwidth-bound and the arithmetic difference is not what is being measured.",
        100.0 * best_exact / peak
    );
    eprintln!(
        "The score pass adds 2*S*4 bytes against 2*S*D for K and V, i.e. 4/D = {:.1}% \
         extra traffic at head_dim {D}.",
        400.0 / D as f64
    );
    // The assertion is on the FAIR twin, not on the flattering baseline.
    assert!(
        tw < 1.15,
        "exact accumulation is {tw:.2}x its own f32 twin -- exactness is supposed to \
         be arithmetically free in this phase, and past ~1.15x the 3% traffic \
         argument no longer carries the cost case. Revisit §6 item 3 rather than \
         quietly reporting the number."
    );
}

/// The device-side control: the f32 baseline in this same file really is
/// nondeterministic under repartition, and the exact one really is not.
///
/// The CPU control in `gpu_attention_invariance.rs` models the flash combine;
/// this runs the actual kernel being timed above, so the cost comparison is
/// against a baseline that demonstrably has the defect exactness removes.
#[test]
#[ignore = "requires a CUDA device"]
fn the_f32_baseline_being_timed_really_does_move_with_the_split_count() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver.");
        return;
    };
    let src = ptx();
    let exact = ctx.load_ptx(&src, "accum_exact").expect("accum_exact");
    let flash = ctx.load_ptx(&src, "accum_f32").expect("accum_f32");

    let (scores, weights, v) = data();

    let d_p = ctx.alloc(B * S * 4).unwrap();
    let d_s = ctx.alloc(B * S * 4).unwrap();
    let d_v = ctx.alloc(B * S * D).unwrap();
    let d_l = ctx.alloc(B * 8).unwrap();
    let d_o = ctx.alloc(B * D * 8).unwrap();
    let bytes32 = |x: &[i32]| x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    ctx.memcpy_htod_at(&d_p, 0, &bytes32(&weights)).unwrap();
    ctx.memcpy_htod_at(&d_s, 0, &bytes32(&scores)).unwrap();
    ctx.memcpy_htod_at(&d_v, 0, &v).unwrap();

    let max_nz = 32usize;
    let d_lp = ctx.alloc(B * max_nz * 4).unwrap();
    let d_mp = ctx.alloc(B * max_nz * 4).unwrap();
    let d_op = ctx.alloc(B * max_nz * D * 4).unwrap();

    let run_exact = |nz: u32| -> Vec<f64> {
        ctx.memset_u8(&d_l, 0).unwrap();
        ctx.memset_u8(&d_o, 0).unwrap();
        let args = vec![
            d_p.device_ptr(),
            d_v.device_ptr(),
            d_l.device_ptr(),
            d_o.device_ptr(),
        ];
        ctx.launch(&exact, (1, B as u32, nz), (D as u32, 1, 1), 0, &args)
            .unwrap();
        ctx.synchronize().unwrap();
        let mut rl = vec![0u8; B * 8];
        let mut ro = vec![0u8; B * D * 8];
        ctx.memcpy_dtoh_at(&mut rl, &d_l, 0).unwrap();
        ctx.memcpy_dtoh_at(&mut ro, &d_o, 0).unwrap();
        let g = |r: &[u8], i: usize| i64::from_le_bytes(r[i * 8..i * 8 + 8].try_into().unwrap());
        let mut out = Vec::with_capacity(B * D);
        for b in 0..B {
            let l = g(&rl, b) as f64;
            for d in 0..D {
                out.push(g(&ro, b * D + d) as f64 / l);
            }
        }
        out
    };

    let run_flash = |nz: u32| -> Vec<f64> {
        let args = vec![
            d_s.device_ptr(),
            d_v.device_ptr(),
            d_lp.device_ptr(),
            d_op.device_ptr(),
            d_mp.device_ptr(),
        ];
        ctx.launch(&flash, (1, B as u32, nz), (D as u32, 1, 1), 0, &args)
            .unwrap();
        ctx.synchronize().unwrap();
        let n = nz as usize;
        let mut rl = vec![0u8; B * n * 4];
        let mut rm = vec![0u8; B * n * 4];
        let mut ro = vec![0u8; B * n * D * 4];
        ctx.memcpy_dtoh_at(&mut rl, &d_lp, 0).unwrap();
        ctx.memcpy_dtoh_at(&mut rm, &d_mp, 0).unwrap();
        ctx.memcpy_dtoh_at(&mut ro, &d_op, 0).unwrap();
        let f = |r: &[u8], i: usize| f32::from_le_bytes(r[i * 4..i * 4 + 4].try_into().unwrap());
        // The flash cross-split combine, exactly as a real kernel does it.
        let mut out = Vec::with_capacity(B * D);
        for b in 0..B {
            let gm = (0..n).fold(f32::NEG_INFINITY, |a, z| a.max(f(&rm, b * n + z)));
            let mut l = 0.0f32;
            let mut acc = vec![0.0f32; D];
            for z in 0..n {
                let c = (f(&rm, b * n + z) - gm).exp2();
                l += f(&rl, b * n + z) * c;
                for d in 0..D {
                    acc[d] += f(&ro, (b * n + z) * D + d) * c;
                }
            }
            out.extend(acc.iter().map(|a| (a / l) as f64));
        }
        out
    };

    let (be, bf) = (run_exact(1), run_flash(1));
    let (mut moved_e, mut moved_f) = (0usize, 0usize);
    for &nz in &[2u32, 4, 8, 16, 32] {
        let (ge, gf) = (run_exact(nz), run_flash(nz));
        moved_e += ge.iter().zip(&be).filter(|(a, b)| a != b).count();
        moved_f += gf.iter().zip(&bf).filter(|(a, b)| a != b).count();
    }
    let total = 5 * B * D;
    eprintln!("exact: {moved_e} of {total} outputs moved with the split count");
    eprintln!("f32  : {moved_f} of {total} outputs moved with the split count");
    assert_eq!(
        moved_e, 0,
        "the exact kernel being timed is not actually split-invariant"
    );
    assert!(
        moved_f > total / 10,
        "the f32 kernel being timed barely moves ({moved_f} of {total}), so it is not \
         a meaningful baseline for what exactness removes"
    );
}
