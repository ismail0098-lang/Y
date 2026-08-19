//! M4 step 1d: **how wrong is the exact path?**
//!
//! `docs/deterministic_inference.md` §6 item 1 is the top kill-criterion —
//! "accuracy loss at int16/int8 operands is unacceptable to the buyer" — and
//! it says to test it early rather than at M5. A full task-accuracy run needs
//! a model and a serving path that do not exist yet (M4 steps 2–3). What CAN
//! be tested now is the part that is a *design choice of this project* rather
//! than of quantization generally:
//!
//! - softmax weights quantised to **Q0.F fixed point** (the kernel uses F=28),
//! - the two-pass structure with a global max instead of an online one.
//!
//! ## The bar
//!
//! Not "is it exact" — it cannot be; `exp` is not exact in any representation.
//! The bar is **is it more wrong than the f32 online softmax that production
//! flash attention already ships?** If the answer is no, the accuracy
//! objection does not attach to determinism, it attaches to quantization, and
//! quantization is the customer's existing decision.
//!
//! Both paths read the **same int8 `V`**, so int8 quantization error is common
//! to them and cancels. This isolates the softmax representation. The size of
//! that common term is reported separately so the two effects can be ranked.
//!
//! ## What is still not covered
//!
//! No model, no calibration, no activation outliers, no per-channel scales.
//! Real LLM quantization is hard because of activation outliers, and these are
//! synthetic score distributions chosen to *include* hard shapes, not observed
//! ones. This bounds the numerics; it does not answer §6 item 1.

use y::fixed_exp::exp2_neg_q16_16;

/// The kernel's fixed-point width for softmax weights.
///
/// **Derived, and the first value was wrong.** F=16 was picked by hand and is
/// catastrophic on an attention sink: with the sink 18 logits above the rest,
/// every non-sink weight is `2^-18 * 2^16 = 0.25`, which rounds to **zero**.
/// The whole tail -- about 1.5% of the softmax mass -- disappears, and the
/// output is 2109x further from an f64 reference than f32 flash attention is.
/// Attention sinks are ubiquitous in real transformers, so this was not an
/// exotic corner.
///
/// The sweep in `the_fixed_point_width_is_wide_enough` shows the error falling
/// monotonically with no knee inside the usable range, so F is set by the
/// RANGE OBLIGATION rather than by a plateau: `p` is at most `2^F` and must fit
/// an i32, `p*v` needs `F + 7` bits, and summing `S` of them needs
/// `F + 7 + log2(S)` bits inside the i64 accumulator. F=28 leaves p at 2^28
/// (three bits of i32 headroom) and supports sequences to 2^28 keys.
const F: u32 = 28;

/// `p*v` is `F + 7` bits (int8 V), and `S` of them need `log2(S)` more.
const MAX_SEQ_BITS: u32 = 63 - F - 7;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in [0, 1).
fn unif(state: &mut u64) -> f64 {
    (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Standard normal, Box-Muller.
fn normal(state: &mut u64) -> f64 {
    let u1 = unif(state).max(1e-300);
    let u2 = unif(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    /// Diffuse attention: nearly uniform weights. Early layers look like this.
    Diffuse,
    /// Ordinary mid-range peaking.
    Moderate,
    /// Sharply peaked - the hard case for a fixed-point weight, because most
    /// weights land near or below the quantisation step.
    Peaked,
    /// An attention sink: one key takes almost all the mass, the rest is
    /// diffuse. Ubiquitous in real transformers and the worst realistic case
    /// for Q0.F.
    Sink,
    /// Heavy-tailed logits, so the dynamic range is far wider than Gaussian.
    HeavyTail,
}

impl Shape {
    fn all() -> [Shape; 5] {
        [
            Shape::Diffuse,
            Shape::Moderate,
            Shape::Peaked,
            Shape::Sink,
            Shape::HeavyTail,
        ]
    }
    /// Logits in base-2 units, matching the kernel's use of `exp2`.
    fn logits(&self, s: usize, st: &mut u64) -> Vec<f64> {
        match self {
            Shape::Diffuse => (0..s).map(|_| normal(st) * 0.5).collect(),
            Shape::Moderate => (0..s).map(|_| normal(st) * 4.0).collect(),
            Shape::Peaked => (0..s).map(|_| normal(st) * 12.0).collect(),
            Shape::Sink => (0..s)
                .map(|i| if i == 0 { 18.0 } else { normal(st) * 2.0 })
                .collect(),
            Shape::HeavyTail => (0..s)
                .map(|_| {
                    let u = unif(st) * 0.98 + 0.01;
                    (std::f64::consts::PI * (u - 0.5)).tan() * 3.0
                })
                .collect(),
        }
    }
}

/// The kernel's temperature: `s` is an exact int32 dot product and `C` folds
/// the softmax scale together with log2(e).
const C: f64 = 0.000_122_070_312_5; // 2^-13

struct Case {
    scores: Vec<i32>,
    v: Vec<i8>,
    d: usize,
}

impl Case {
    fn new(shape: Shape, s: usize, d: usize, st: &mut u64) -> Case {
        // Generated in logit space, then pushed back through the integer
        // pipeline the kernel actually sees.
        let scores = shape
            .logits(s, st)
            .iter()
            .map(|x| (x / C).round() as i32)
            .collect();
        let v = (0..s * d)
            .map(|_| (unif(st) * 255.0 - 127.0).round().clamp(-127.0, 127.0) as i8)
            .collect();
        Case { scores, v, d }
    }

    /// f64 softmax over the same int scores and the same int8 V. This is the
    /// thing both candidates are wrong relative to.
    fn reference(&self) -> Vec<f64> {
        let m = *self.scores.iter().max().unwrap();
        let w: Vec<f64> = self
            .scores
            .iter()
            .map(|&s| ((s - m) as f64 * C).exp2())
            .collect();
        let l: f64 = w.iter().sum();
        (0..self.d)
            .map(|d| {
                w.iter()
                    .enumerate()
                    .map(|(i, &p)| p * self.v[i * self.d + d] as f64)
                    .sum::<f64>()
                    / l
            })
            .collect()
    }

    /// The exact path: global max, Q0.f weights, integer accumulation.
    fn exact(&self, f: u32) -> Vec<f64> {
        let scale = (1u64 << f) as f32;
        let m = *self.scores.iter().max().unwrap();
        // f32 exp, as the kernel does. `ex2.approx.f32` is a further ~2 ulp
        // looser than this, which is far below the quantisation step.
        let p: Vec<i64> = self
            .scores
            .iter()
            .map(|&s| (((s - m) as f32 * C as f32).exp2() * scale).round() as i64)
            .collect();
        let l: i64 = p.iter().sum();
        assert!(l > 0, "every weight quantised to zero at F={f}");
        (0..self.d)
            .map(|d| {
                p.iter()
                    .enumerate()
                    .map(|(i, &w)| w * self.v[i * self.d + d] as i64)
                    .sum::<i64>() as f64
                    / l as f64
            })
            .collect()
    }

    /// The exact path with the **integer** exp of `src/fixed_exp.rs` instead
    /// of `ex2.approx.f32`. This is what makes the result identical on two
    /// different GPUs, not merely on two launches of the same one.
    ///
    /// The conversion into the exp's Q16.16 argument is a SHIFT, not a
    /// multiply: `t = (m - s) * C` with `C = 2^-13`, so as Q16.16 it is
    /// `(m - s) << 3`. There is no float anywhere on this path.
    fn exact_int_exp(&self) -> Vec<f64> {
        let m = *self.scores.iter().max().unwrap();
        let p: Vec<i64> = self
            .scores
            .iter()
            .map(|&s| {
                let dt = (m - s) as u32; // >= 0 by construction
                exp2_neg_q16_16(dt.saturating_mul(8)) as i64
            })
            .collect();
        let l: i64 = p.iter().sum();
        assert!(l > 0, "every weight quantised to zero under the integer exp");
        (0..self.d)
            .map(|d| {
                p.iter()
                    .enumerate()
                    .map(|(i, &w)| w * self.v[i * self.d + d] as i64)
                    .sum::<i64>() as f64
                    / l as f64
            })
            .collect()
    }

    /// The production baseline: f32 online softmax, rescaled once per tile,
    /// exactly as a flash kernel carries it.
    fn flash_f32(&self, tile: usize) -> Vec<f64> {
        let (mut m, mut l) = (f32::NEG_INFINITY, 0.0f32);
        let mut acc = vec![0.0f32; self.d];
        for chunk in self.scores.chunks(tile) {
            let xs: Vec<f32> = chunk.iter().map(|&s| s as f32 * C as f32).collect();
            let tm = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mn = m.max(tm);
            let corr = (m - mn).exp2();
            l *= corr;
            for a in acc.iter_mut() {
                *a *= corr;
            }
            for (j, &x) in xs.iter().enumerate() {
                let p = (x - mn).exp2();
                l += p;
                let base = (chunk.as_ptr() as usize - self.scores.as_ptr() as usize)
                    / std::mem::size_of::<i32>();
                for d in 0..self.d {
                    acc[d] += p * self.v[(base + j) * self.d + d] as f32;
                }
            }
            m = mn;
        }
        acc.iter().map(|a| (a / l) as f64).collect()
    }
}

fn worst(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

#[test]
fn the_exact_path_is_no_more_wrong_than_the_f32_flash_path() {
    const D: usize = 128;
    let mut st = 0x5EED_1234_ABCD_0001u64;

    println!("Output values lie in +-127. Error vs an f64 softmax over the same int8 V.");
    println!(
        "{:>10} {:>6} {:>13} {:>13} {:>13}  {:>7} {:>7}",
        "shape", "seq", format!("exact f32exp"), "exact intexp", "f32 flash", "f32/fl", "int/fl"
    );

    let mut worst_ratio = 0.0f64;
    let mut worst_int_ratio = 0.0f64;
    let mut any_flash_error = 0.0f64;
    for shape in Shape::all() {
        for &s in &[128usize, 1024, 4096] {
            let case = Case::new(shape, s, D, &mut st);
            let r = case.reference();
            let e = worst(&case.exact(F), &r);
            let ei = worst(&case.exact_int_exp(), &r);
            let f = worst(&case.flash_f32(64), &r);
            any_flash_error = any_flash_error.max(f);
            let ratio = if f > 0.0 { e / f } else { f64::INFINITY };
            let ratio_i = if f > 0.0 { ei / f } else { f64::INFINITY };
            worst_ratio = worst_ratio.max(ratio);
            worst_int_ratio = worst_int_ratio.max(ratio_i);
            println!(
                "{shape:>10?} {s:>6} {e:>13.3e} {ei:>13.3e} {f:>13.3e}  {ratio:>6.2}x {ratio_i:>6.2}x"
            );
        }
    }

    // The control: if the f32 baseline were itself exact, "no worse than f32"
    // would be a claim about nothing.
    assert!(
        any_flash_error > 0.0,
        "the f32 flash baseline showed zero error against f64, so this comparison \
         is vacuous"
    );
    println!();
    println!("worst exact/flash error ratio, f32 exp    : {worst_ratio:.2}x");
    println!("worst exact/flash error ratio, integer exp: {worst_int_ratio:.2}x");
    // The integer exp is the shippable one -- it is the only version that is
    // identical across GPU architectures -- so it carries the same bar.
    assert!(
        worst_int_ratio < 1.0,
        "the integer-exp path is {worst_int_ratio:.2}x the f32 flash path's error. \
         It is the only version that is architecture-independent, so if it cannot \
         meet the accuracy bar the cross-GPU claim costs accuracy and must say so."
    );
    // Strictly better than the baseline, not merely close to it. F=16 read
    // 2109x here; anything above 1.0 means the design is the accuracy story.
    assert!(
        worst_ratio < 1.0,
        "the exact path is {worst_ratio:.2}x the f32 flash path's error. It is \
         supposed to be BETTER, since it accumulates exactly and flash does not. \
         Above 1.0 the accuracy regression is attributable to THIS design (Q0.{F} \
         softmax weights) rather than to quantization, and §6 item 1 lands on us."
    );
}

/// F = 16 was picked by hand. This is what justifies it, and what shows where
/// it stops working.
#[test]
fn the_fixed_point_width_is_wide_enough_and_the_sweep_shows_why() {
    const D: usize = 128;
    let mut st = 0xC0FF_EE00_1234_5678u64;

    println!("Worst error vs f64, by fixed-point width, over all shapes at seq 1024:");
    println!("{:>4}  {:>12}  {:>12}", "F", "worst err", "vs f32 flash");

    let cases: Vec<Case> = Shape::all()
        .iter()
        .map(|&sh| Case::new(sh, 1024, D, &mut st))
        .collect();
    let refs: Vec<Vec<f64>> = cases.iter().map(|c| c.reference()).collect();
    let flash = cases
        .iter()
        .zip(&refs)
        .map(|(c, r)| worst(&c.flash_f32(64), r))
        .fold(0.0f64, f64::max);

    let mut prev = f64::INFINITY;
    let mut errs = Vec::new();
    for f in [8u32, 12, 16, 20, 24, 26, 28, 30] {
        let e = cases
            .iter()
            .zip(&refs)
            .map(|(c, r)| worst(&c.exact(f), r))
            .fold(0.0f64, f64::max);
        println!("{f:>4}  {e:>12.3e}  {:>11.2}x", e / flash);
        errs.push((f, e));
        // Every extra bit must actually buy something until the floor is hit,
        // or the parameter is not doing what its name says.
        assert!(
            e <= prev * 1.05 || f >= 24,
            "widening the fixed point from below {f} bits made the error WORSE \
             ({prev:.3e} -> {e:.3e}); the quantisation is not behaving monotonically"
        );
        prev = e;
    }

    let g = |w: u32| errs.iter().find(|(f, _)| *f == w).unwrap().1;
    println!();
    println!(
        "F=8 -> {:.3e},  F=16 -> {:.3e},  F={F} -> {:.3e},  F=30 -> {:.3e}",
        g(8),
        g(16),
        g(F),
        g(30)
    );
    println!(
        "no knee: the error is still falling at F=30, so F is set by the range \
         obligation, not by a plateau."
    );

    // The value actually shipped must beat the baseline it replaces...
    assert!(
        g(F) < flash,
        "the shipped F={F} ({:.3e}) is worse than f32 flash ({flash:.3e})",
        g(F)
    );
    // ...and the previously-shipped 16 must NOT, or this correction was
    // unnecessary and the sweep is not measuring what it claims.
    assert!(
        g(16) > flash * 10.0,
        "F=16 ({:.3e}) is not dramatically worse than f32 flash ({flash:.3e}); the \
         regression this file was written to catch would not have been caught",
        g(16)
    );
    // The range obligation is arithmetic, and stating it is the whole reason F
    // can be chosen at all.
    assert!(
        F + 7 + MAX_SEQ_BITS <= 63,
        "F={F} with int8 V overflows the i64 accumulator before 2^{MAX_SEQ_BITS} keys"
    );
    assert!(
        (1u64 << F) < i32::MAX as u64,
        "p = 2^{F} does not fit the i32 register the kernel computes it in"
    );
    println!(
        "range obligation: p <= 2^{F} fits i32; p*v is {} bits; sequences to 2^{MAX_SEQ_BITS} keys fit i64.",
        F + 7
    );
}

/// Rank the two error sources, so the reader knows which one to care about.
///
/// If int8 `V` — the customer's own quantization decision, common to both
/// paths — dominates the softmax representation, then arguing about Q0.16 is
/// arguing about the small term.
#[test]
fn int8_v_dominates_the_softmax_representation() {
    const D: usize = 128;
    let mut st = 0xABCD_0000_9999_4321u64;

    let mut worst_softmax = 0.0f64;
    let mut worst_v = 0.0f64;
    for shape in Shape::all() {
        let case = Case::new(shape, 1024, D, &mut st);
        let r = case.reference();
        worst_softmax = worst_softmax.max(worst(&case.exact(F), &r));

        // The same attention, but with V perturbed by a quantisation step -
        // i.e. what int8 itself costs, independent of any of this.
        let mut jittered = Case {
            scores: case.scores.clone(),
            v: case.v.clone(),
            d: D,
        };
        for x in jittered.v.iter_mut() {
            *x = x.saturating_add(if (*x as i32) % 2 == 0 { 1 } else { -1 });
        }
        worst_v = worst_v.max(worst(&jittered.reference(), &r));
    }
    println!("worst error from Q0.{F} softmax weights : {worst_softmax:.3e}");
    println!("worst error from one int8 step in V     : {worst_v:.3e}");
    println!("ratio (V / softmax)                     : {:.1}x", worst_v / worst_softmax);
    assert!(
        worst_v > worst_softmax,
        "the softmax representation ({worst_softmax:.3e}) costs MORE than a single \
         int8 step in V ({worst_v:.3e}). That inverts the argument in this file's \
         header -- the design's own error would then be the dominant term and \
         would need defending on its own."
    );
}

/// This file models the kernel with three constants, and until now it owned all
/// three privately.
///
/// `F`, `C` and `MAX_SEQ_BITS` are transcriptions of numbers that live in
/// `src/fixed_exp.rs` and `src/exact_attention.rs`. Nothing tied them together,
/// so a change on either side would leave this file certifying an accuracy
/// figure for a design the compiler no longer emits — the tests would stay
/// green and stop being about the kernel.
///
/// That is not hypothetical for `C`. The kernel's temperature became a runtime
/// parameter, and its multiplier `KFix` is `C * 2^32`, one factor of `2^16`
/// larger than the multiplier the torch demo uses, because the kernel shifts.
/// `tools/ptx_bridge.py` passed the demo's, and got a uniform softmax its own
/// differential could not see.
#[test]
fn the_model_here_is_the_kernel_the_compiler_emits() {
    // The Q0.F weight scale is not a free parameter of this file: it is the
    // output scale of the exp the kernel calls.
    assert_eq!(
        F,
        y::fixed_exp::EXP2_FRAC_BITS,
        "F here must be the exp's own output scale; `exact_int_exp` calls \
         `exp2_neg_q16_16` directly, so if these diverge the sweep tests one \
         width while the shipped path uses another"
    );

    // The range obligation stated in `the_fixed_point_width_is_wide_enough`
    // must be no weaker than the bound the generator enforces.
    assert!(
        (1usize << MAX_SEQ_BITS) <= y::exact_attention::MAX_EXACT_SEQ_LEN,
        "this file allows 2^{MAX_SEQ_BITS} keys but `attention_ptx` refuses \
         past {}; the accuracy argument here would then cover sequences the \
         kernel will not compile",
        y::exact_attention::MAX_EXACT_SEQ_LEN
    );

    // `C` is the log2-units-per-score-unit this file's cases are built in, and
    // `exact_int_exp` converts with `<< 3`. The kernel reaches the same place
    // through `((m - s) * KFix + 2^15) >> 16`, so the two agree only when
    // `KFix == C * 2^32`. The synthetic tests pass `8 << 16`.
    let legacy_kfix = (8i64 << 16) as f64;
    assert_eq!(
        C,
        legacy_kfix / 2f64.powi(32),
        "C must be the temperature the kernel's `8 << 16` denotes"
    );
    for ds in [0i64, 1, 97, 4095] {
        assert_eq!(
            ((ds * (8 << 16)) + 32768) >> 16,
            ds << 3,
            "the kernel's multiply-and-shift must reproduce this file's `<< 3`"
        );
    }
}
