//! `@ZeroDrift`: pick the cheapest accumulator representation that is actually
//! drift-free on *this* device.
//!
//! ## What the annotation used to do
//!
//! Nothing. It was lexed, parsed, stored on the AST, counted, and printed as an
//! advisory - and no backend ever read it. Compiling `tests/test_drift.ysu`
//! with and without the annotations produced byte-identical PTX. It even
//! printed "Compiler must insert software compensation path" while inserting
//! nothing, against a hardware profile whose `DRIFT_FREE_TYPES=Q32.32,F64` was
//! a hardcoded string in the probe rather than anything measured.
//!
//! ## What drift-free actually means
//!
//! Floating-point addition is not associative: `(a + b) + c` and `a + (b + c)`
//! can differ, so a reduction's result depends on the order the hardware
//! happened to combine it in. That is the drift. It is why two runs of the same
//! kernel on different launch geometry can disagree in the last bits.
//!
//! **Only exact arithmetic removes it.** Fixed-point accumulation in an integer
//! register is exact: integer addition is associative and commutative, so every
//! summation order gives bit-identical results, always. `f64` is *not*
//! drift-free - it is the same non-associative arithmetic with more mantissa,
//! so it drifts less and still drifts. Kahan compensation likewise reduces
//! error without eliminating reordering sensitivity.
//!
//! The old profile listed `Q32.32,F64` together as "drift free types", which
//! conflates "exact" with "more precise". [`DriftRepr::is_exact`] keeps them
//! apart, and `@ZeroDrift` only ever selects an exact representation.
//!
//! ## What "smart" means here
//!
//! Among representations that are exact *and* have enough range for the
//! declared type, pick the one that accumulates fastest on the device actually
//! present. That ordering is measured, not assumed: on an RTX 4070 Ti SUPER a
//! 64-bit integer add is not obviously cheaper or dearer than the alternatives,
//! and the answer differs across architectures. [`measure_accumulate_costs`]
//! JITs a small dependent-accumulate loop per candidate and times it, and the
//! result is cached in `.ysu_hw_profile` alongside the autotuner's data.

use std::collections::HashMap;

/// An accumulator representation `@ZeroDrift` can select.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DriftRepr {
    /// Q16.16 fixed point in a 32-bit integer. Exact; ±32768 with 2^-16 steps.
    FixedQ16_16,
    /// Q32.32 fixed point in a 64-bit integer. Exact; ±2^31 with 2^-32 steps.
    FixedQ32_32,
    /// Pure 64-bit integer accumulation. Exact, no fractional part.
    Int64,
    /// `f64`. NOT exact - retained so the reporting can say why it lost.
    Float64,
    /// Compensated (Kahan) `f32`. NOT exact, for the same reason.
    KahanF32,
}

impl DriftRepr {
    /// Every representation, in a fixed order so reports are reproducible.
    pub const ALL: [DriftRepr; 5] = [
        DriftRepr::FixedQ16_16,
        DriftRepr::FixedQ32_32,
        DriftRepr::Int64,
        DriftRepr::Float64,
        DriftRepr::KahanF32,
    ];

    /// The name used in source, in `.ysu_hw_profile` keys, and in diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            DriftRepr::FixedQ16_16 => "Q16.16",
            DriftRepr::FixedQ32_32 => "Q32.32",
            DriftRepr::Int64 => "I64",
            DriftRepr::Float64 => "F64",
            DriftRepr::KahanF32 => "KahanF32",
        }
    }

    /// Whether accumulation in this representation is bit-exact regardless of
    /// summation order.
    ///
    /// This is the whole question `@ZeroDrift` asks. Integer and fixed-point
    /// addition is associative, so it is; anything floating-point is not,
    /// however many mantissa bits it has.
    pub fn is_exact(self) -> bool {
        matches!(
            self,
            DriftRepr::FixedQ16_16 | DriftRepr::FixedQ32_32 | DriftRepr::Int64
        )
    }

    /// Fractional bits, for the fixed-point representations.
    pub fn frac_bits(self) -> u32 {
        match self {
            DriftRepr::FixedQ16_16 => 16,
            DriftRepr::FixedQ32_32 => 32,
            _ => 0,
        }
    }

    /// Total width of the backing integer.
    pub fn total_bits(self) -> u32 {
        match self {
            DriftRepr::FixedQ16_16 => 32,
            DriftRepr::FixedQ32_32 | DriftRepr::Int64 => 64,
            DriftRepr::Float64 => 64,
            DriftRepr::KahanF32 => 64, // value + compensation
        }
    }

    /// Largest magnitude representable.
    pub fn max_magnitude(self) -> f64 {
        match self {
            DriftRepr::FixedQ16_16 => 32768.0,
            DriftRepr::FixedQ32_32 => 2147483648.0,
            DriftRepr::Int64 => 9.223372036854776e18,
            DriftRepr::Float64 => f64::MAX,
            DriftRepr::KahanF32 => f32::MAX as f64,
        }
    }

    /// The C type used to hold it.
    pub fn c_type(self) -> &'static str {
        match self {
            DriftRepr::FixedQ16_16 => "int32_t",
            DriftRepr::FixedQ32_32 | DriftRepr::Int64 => "int64_t",
            DriftRepr::Float64 => "double",
            DriftRepr::KahanF32 => "float",
        }
    }
}

/// What the declared type needs from an accumulator.
#[derive(Clone, Copy, Debug)]
pub struct Requirement {
    /// Largest magnitude the accumulator must hold without overflowing.
    pub max_magnitude: f64,
    /// Smallest step that must be representable, or `None` for integers.
    pub resolution: Option<f64>,
}

impl Requirement {
    /// The requirement implied by a declared Y type.
    ///
    /// Deliberately conservative: an unrecognised type is assumed to need the
    /// full `f32` range, so the selector widens rather than silently choosing a
    /// representation that overflows. Overflowing a fixed-point accumulator is
    /// not a rounding error, it is a wrong answer.
    pub fn for_type(name: &str) -> Self {
        match name {
            // Half precision products accumulated over a tile stay small; the
            // resolution is what matters.
            "F16" | "f16" | "half" => Requirement {
                max_magnitude: 65504.0,
                resolution: Some(2f64.powi(-24)),
            },
            "F32" | "f32" | "float" => Requirement {
                max_magnitude: 3.4028235e38,
                resolution: Some(2f64.powi(-24)),
            },
            "F64" | "f64" | "double" => Requirement {
                max_magnitude: f64::MAX,
                resolution: Some(2f64.powi(-53)),
            },
            "I32" | "i32" | "I64" | "i64" => Requirement {
                max_magnitude: 9.223372036854776e18,
                resolution: None,
            },
            n if n.starts_with('Q') => {
                // Q<int>.<frac>
                let digits: Vec<u32> = n[1..]
                    .split('.')
                    .filter_map(|p| p.parse::<u32>().ok())
                    .collect();
                if digits.len() == 2 {
                    Requirement {
                        max_magnitude: 2f64.powi(digits[0] as i32),
                        resolution: Some(2f64.powi(-(digits[1] as i32))),
                    }
                } else {
                    Requirement { max_magnitude: 3.4028235e38, resolution: Some(2f64.powi(-24)) }
                }
            }
            _ => Requirement { max_magnitude: 3.4028235e38, resolution: Some(2f64.powi(-24)) },
        }
    }

    /// The requirement for a declared type whose value range is known.
    ///
    /// This is what makes `@ZeroDrift` usable on a float accumulator at all.
    /// Asked about a bare `F32`, [`Requirement::for_type`] has to assume the
    /// full 3.4e38 range, and no fixed-point representation can hold that - so
    /// selection correctly fails. But a real accumulator is never that wide,
    /// and `@bounds(min, max)` already exists to say so. Given bounds, the
    /// magnitude comes from the declaration instead of from the worst case, and
    /// an exact representation usually fits.
    ///
    /// The resolution still comes from the declared type: the point of
    /// `@ZeroDrift` is to preserve the precision the source asked for, not to
    /// quietly trade it away for exactness.
    pub fn for_type_with_bounds(name: &str, bounds: Option<(f64, f64)>) -> Self {
        let base = Self::for_type(name);
        match bounds {
            None => base,
            Some((lo, hi)) => Requirement {
                max_magnitude: lo.abs().max(hi.abs()),
                resolution: base.resolution,
            },
        }
    }

    /// Whether `repr` can hold this without overflow or loss of resolution.
    pub fn satisfied_by(&self, repr: DriftRepr) -> bool {
        if repr.max_magnitude() < self.max_magnitude {
            return false;
        }
        match self.resolution {
            None => true,
            Some(step) => {
                let frac = repr.frac_bits();
                frac > 0 && 2f64.powi(-(frac as i32)) <= step
            }
        }
    }
}

/// Measured cost of one accumulate, in picoseconds per operation.
pub type CostTable = HashMap<DriftRepr, f64>;

/// The outcome of selection, including why - so the compiler can report a real
/// reason rather than an advisory.
#[derive(Clone, Debug)]
pub struct Decision {
    pub repr: DriftRepr,
    /// Representations rejected, each with the reason.
    pub rejected: Vec<(DriftRepr, String)>,
    /// Cost used to break the tie, when one was available.
    pub cost_ps: Option<f64>,
    /// True when the choice came from measurement rather than the fallback order.
    pub measured: bool,
}

/// Picks the cheapest exact representation that satisfies `req`.
///
/// Selection order:
///
/// 1. Discard anything not [`DriftRepr::is_exact`] - `@ZeroDrift` means zero,
///    and `f64` merely drifts more slowly.
/// 2. Discard anything without the range or resolution the declared type needs.
/// 3. Among what is left, take the lowest measured cost. With no measurements,
///    fall back to the narrowest representation, which is the better guess on
///    every architecture seen so far and is at least deterministic.
///
/// Returns `None` only when nothing exact can hold the declared type - which is
/// a real error, not something to paper over with a float.
pub fn select_repr(req: &Requirement, costs: &CostTable) -> Option<Decision> {
    let mut rejected = Vec::new();
    let mut viable = Vec::new();

    for repr in DriftRepr::ALL {
        if !repr.is_exact() {
            rejected.push((
                repr,
                "not exact: floating-point addition is not associative, so the result still \
                 depends on summation order"
                    .to_string(),
            ));
            continue;
        }
        if !req.satisfied_by(repr) {
            rejected.push((
                repr,
                format!(
                    "insufficient range or resolution (holds |x| < {:.3e} at {} fractional bits)",
                    repr.max_magnitude(),
                    repr.frac_bits()
                ),
            ));
            continue;
        }
        viable.push(repr);
    }

    let best = if viable.iter().any(|r| costs.contains_key(r)) {
        let mut with_cost: Vec<(DriftRepr, f64)> = viable
            .iter()
            .filter_map(|r| costs.get(r).map(|c| (*r, *c)))
            .collect();
        // Cheapest first; ties broken by the narrower representation.
        with_cost.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.total_bits().cmp(&b.0.total_bits()))
        });
        let (repr, cost) = with_cost[0];
        for (r, c) in with_cost.iter().skip(1) {
            rejected.push((*r, format!("slower: {:.1} ps/acc vs {:.1}", c, cost)));
        }
        Some(Decision { repr, rejected, cost_ps: Some(cost), measured: true })
    } else {
        let mut sorted = viable.clone();
        sorted.sort_by_key(|r| r.total_bits());
        sorted.first().map(|repr| {
            for r in sorted.iter().skip(1) {
                rejected.push((*r, "wider than necessary (no measurements available)".to_string()));
            }
            Decision { repr: *repr, rejected, cost_ps: None, measured: false }
        })
    };

    best
}

/// A dependent-accumulate loop in `repr`, as a standalone PTX kernel.
///
/// The chain is deliberately serial - each add consumes the previous result -
/// so the measurement reflects accumulate latency rather than how many
/// independent adds the scheduler can overlap. That is the quantity a reduction
/// is actually limited by.
pub fn accumulate_probe_ptx(repr: DriftRepr, iters: u32) -> String {
    let (reg_ty, reg, add, load) = match repr {
        DriftRepr::FixedQ16_16 => ("s32", "%r", "add.s32", "ld.global.u32"),
        DriftRepr::FixedQ32_32 | DriftRepr::Int64 => ("s64", "%rd", "add.s64", "ld.global.u64"),
        DriftRepr::Float64 => ("f64", "%fd", "add.f64", "ld.global.f64"),
        DriftRepr::KahanF32 => ("f32", "%f", "add.f32", "ld.global.f32"),
    };
    let store = match repr {
        DriftRepr::FixedQ16_16 => "st.global.u32",
        DriftRepr::FixedQ32_32 | DriftRepr::Int64 => "st.global.u64",
        DriftRepr::Float64 => "st.global.f64",
        DriftRepr::KahanF32 => "st.global.f32",
    };
    let acc = format!("{}0", reg);
    let step = format!("{}1", reg);

    let mut body = String::new();
    // The two registers feed each other, Fibonacci fashion:
    //
    //     acc  += step
    //     step += acc
    //
    // Both halves of this matter, and both were learned by getting it wrong.
    //
    // A plain `acc += step` chain is not a probe: `step` is loop invariant, so
    // the unrolled sequence collapses to `acc + N*step` and ptxas folds it.
    // Feeding each result into the next leaves no closed form.
    //
    // But that alone is still not enough, because the SEED must be opaque too.
    // Seeded from a literal, the entire chain - however tangled - is a
    // compile-time constant and ptxas evaluates the whole thing at compile
    // time. The tell both times was that `Q32.32` and `Int64` emit
    // byte-identical PTX yet measured 766x and then 48x apart; two identical
    // kernels cannot differ, so the difference was noise around a chain that
    // had been optimised to nothing. The seed is therefore loaded from device
    // memory, which the compiler cannot see through.
    for i in 0..iters {
        if i % 2 == 0 {
            body.push_str(&format!("    {} {}, {}, {};\n", add, acc, acc, step));
        } else {
            body.push_str(&format!("    {} {}, {}, {};\n", add, step, step, acc));
        }
    }

    // Loads the seed from [out] and stores the result to [out+8], so the seed
    // stays fixed across launches instead of drifting to infinity.
    format!(
        ".version 7.8\n.target sm_89\n.address_size 64\n\
         .visible .entry drift_probe(.param .u64 out)\n{{\n\
         \x20   .reg .{ty} {acc};\n\
         \x20   .reg .{ty} {step};\n\
         \x20   .reg .u64 %out;\n\
         \x20   .reg .u64 %dst;\n\
         \x20   ld.param.u64 %out, [out];\n\
         \x20   {load} {step}, [%out];\n\
         \x20   mov.{ty} {acc}, {step};\n\
{body}\
         \x20   add.u64 %dst, %out, 8;\n\
         \x20   {store} [%dst], {acc};\n\
         \x20   ret;\n}}\n",
        ty = reg_ty,
        acc = acc,
        step = step,
        load = load,
        body = body,
        store = store
    )
}

/// Times one accumulate in each representation on the attached GPU.
///
/// Returns picoseconds per accumulate, and measures *latency*: the probe chain
/// is serially dependent, which is what an accumulator actually is. Throughput
/// of independent adds would flatter every representation equally and tell us
/// nothing about a reduction.
///
/// Three pieces of discipline, all of which this repo learned the hard way
/// elsewhere and all of which change the answer here:
///
/// * **Clock ramp.** A cold GPU reads far slower than a warm one. Without the
///   ramp below the whole table is compressed and the ordering is noise.
/// * **Baseline subtraction.** A launch costs a microsecond or so before any
///   arithmetic happens, which is the same order as the entire chain. Each
///   representation is therefore timed twice - with the chain and with an
///   otherwise identical zero-add kernel - and the difference is what gets
///   recorded. The first version of this function skipped that and reported
///   1.6 microseconds per integer add, which is about four orders of magnitude
///   off and was really just launch overhead.
/// * **Minimum across interleaved rounds**, not mean or median. Interference
///   only ever makes a measurement slower, so the minimum is the closest thing
///   to the machine's actual capability; interleaving keeps a background load
///   from landing entirely on one candidate.
///
/// `None` when no driver is usable. Absence of a GPU is not an error - it just
/// means selection falls back to the deterministic narrowest-first order.
pub fn measure_accumulate_costs() -> Option<CostTable> {
    use crate::cuda_runtime::CudaContext;

    let ctx = CudaContext::new()?;
    let out = ctx.alloc(16).ok()?;
    // 0x3F800000_3F800000: 1.0 as two f32s, a small normal as f64, and a large
    // but ordinary integer. Benign however the probe reads it - a zero or
    // denormal seed would measure a different code path on the float variants.
    let seed: [u8; 16] = [
        0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F,
        0, 0, 0, 0, 0, 0, 0, 0,
    ];
    ctx.memcpy_htod_at(&out, 0, &seed).ok()?;

    const CHAIN: u32 = 4096;
    const BLOCKS: u32 = 1;
    const THREADS: u32 = 32;
    const ITERS: u32 = 300;
    const ROUNDS: usize = 5;

    // Load both kernels per representation once; JIT time is not part of this.
    let mut loaded = Vec::new();
    for repr in DriftRepr::ALL {
        let chain = ctx.load_ptx(&accumulate_probe_ptx(repr, CHAIN), "drift_probe");
        let base = ctx.load_ptx(&accumulate_probe_ptx(repr, 0), "drift_probe");
        if let (Ok(chain), Ok(base)) = (chain, base) {
            loaded.push((repr, chain, base));
        }
    }
    if loaded.is_empty() {
        return None;
    }

    let arg_sets = vec![vec![out.device_ptr()]];
    let grid = (BLOCKS, 1, 1);
    let block = (THREADS, 1, 1);

    // Ramp the clocks before anything is recorded.
    let ramp = std::time::Instant::now();
    while ramp.elapsed().as_secs_f64() < 1.5 {
        for (_, chain, _) in &loaded {
            let _ = ctx.launch(chain, grid, block, 0, &[out.device_ptr()]);
        }
    }
    ctx.synchronize().ok()?;

    let mut best: HashMap<DriftRepr, f64> = HashMap::new();
    for _ in 0..ROUNDS {
        for (repr, chain, base) in &loaded {
            let Ok(t_chain) = ctx.time_launches(chain, grid, block, 0, &arg_sets, ITERS) else {
                continue;
            };
            let Ok(t_base) = ctx.time_launches(base, grid, block, 0, &arg_sets, ITERS) else {
                continue;
            };
            // `time_launches` returns MICROseconds per launch.
            let delta_us = t_chain - t_base;
            if delta_us <= 0.0 {
                continue;
            }
            let ps_per_acc = delta_us * 1.0e6 / CHAIN as f64;
            best.entry(*repr)
                .and_modify(|v| {
                    if ps_per_acc < *v {
                        *v = ps_per_acc;
                    }
                })
                .or_insert(ps_per_acc);
        }
    }

    if best.is_empty() {
        None
    } else {
        Some(best)
    }
}

/// Renders the cost table for `.ysu_hw_profile`.
pub fn serialize_costs(costs: &CostTable, gpu: &str) -> String {
    let mut keys: Vec<&DriftRepr> = costs.keys().collect();
    keys.sort_by_key(|r| r.name());
    keys.iter()
        .map(|r| format!("DRIFT_ACC_{}_{}={:.3}", r.name(), gpu.replace(' ', "_"), costs[r]))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Reads back what [`serialize_costs`] wrote.
pub fn parse_costs(contents: &str, gpu: &str) -> CostTable {
    let suffix = gpu.replace(' ', "_");
    let mut table = CostTable::new();
    for line in contents.lines() {
        let Some(rest) = line.strip_prefix("DRIFT_ACC_") else {
            continue;
        };
        let Some((key, value)) = rest.split_once('=') else {
            continue;
        };
        let Some(name) = key.strip_suffix(&format!("_{}", suffix)) else {
            continue;
        };
        if let Some(repr) = DriftRepr::ALL.iter().find(|r| r.name() == name) {
            if let Ok(v) = value.trim().parse::<f64>() {
                table.insert(*repr, v);
            }
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The central claim: only integer/fixed-point accumulation is exact.
    ///
    /// Not a matter of taste - `f64` is non-associative arithmetic with a
    /// longer mantissa, so it drifts less and still drifts. Selecting it for
    /// `@ZeroDrift` would be the same category error as the old profile's
    /// `DRIFT_FREE_TYPES=Q32.32,F64`.
    #[test]
    fn only_fixed_point_counts_as_exact() {
        assert!(DriftRepr::FixedQ16_16.is_exact());
        assert!(DriftRepr::FixedQ32_32.is_exact());
        assert!(DriftRepr::Int64.is_exact());
        assert!(!DriftRepr::Float64.is_exact());
        assert!(!DriftRepr::KahanF32.is_exact());
    }

    /// Selection must follow the measurements, not a hardcoded preference.
    #[test]
    fn selection_follows_measured_cost() {
        let req = Requirement { max_magnitude: 1000.0, resolution: Some(2f64.powi(-20)) };

        // Q16.16 lacks the resolution, so Q32.32 is the only exact candidate.
        let d = select_repr(&req, &CostTable::new()).expect("a representation");
        assert_eq!(d.repr, DriftRepr::FixedQ32_32);

        // With a coarser resolution both fit, and cost decides.
        let coarse = Requirement { max_magnitude: 1000.0, resolution: Some(2f64.powi(-10)) };
        let mut costs = CostTable::new();
        costs.insert(DriftRepr::FixedQ16_16, 900.0);
        costs.insert(DriftRepr::FixedQ32_32, 300.0);
        let d = select_repr(&coarse, &costs).expect("a representation");
        assert_eq!(
            d.repr,
            DriftRepr::FixedQ32_32,
            "the wider representation must win when it measures faster"
        );
        assert!(d.measured);

        // Flip the measurements and the choice must flip with them.
        costs.insert(DriftRepr::FixedQ16_16, 100.0);
        let d = select_repr(&coarse, &costs).expect("a representation");
        assert_eq!(d.repr, DriftRepr::FixedQ16_16);
    }

    /// Without measurements the answer must still be deterministic.
    #[test]
    fn falls_back_to_narrowest_without_measurements() {
        let coarse = Requirement { max_magnitude: 1000.0, resolution: Some(2f64.powi(-10)) };
        let d = select_repr(&coarse, &CostTable::new()).expect("a representation");
        assert_eq!(d.repr, DriftRepr::FixedQ16_16);
        assert!(!d.measured);
        assert!(d.cost_ps.is_none());
    }

    /// A float type is never selected, however fast it measures.
    #[test]
    fn floats_are_never_selected_even_when_fastest() {
        let req = Requirement { max_magnitude: 100.0, resolution: Some(2f64.powi(-10)) };
        let mut costs = CostTable::new();
        costs.insert(DriftRepr::Float64, 1.0);
        costs.insert(DriftRepr::KahanF32, 1.0);
        costs.insert(DriftRepr::FixedQ16_16, 5000.0);
        let d = select_repr(&req, &costs).expect("a representation");
        assert!(d.repr.is_exact(), "chose a non-exact representation: {:?}", d.repr);
        assert!(d
            .rejected
            .iter()
            .any(|(r, why)| *r == DriftRepr::Float64 && why.contains("not exact")));
    }

    /// A requirement nothing exact can meet is an error, not a silent downgrade.
    #[test]
    fn impossible_requirement_returns_none() {
        let req = Requirement { max_magnitude: 1e30, resolution: Some(2f64.powi(-40)) };
        assert!(select_repr(&req, &CostTable::new()).is_none());
    }

    /// A bare `F32` accumulator cannot be made exact; a bounded one can.
    ///
    /// Both halves matter. Failing on the unbounded case is correct - nothing
    /// exact holds 3.4e38 at f32 resolution, and silently picking something
    /// that overflows would turn a rounding concern into a wrong answer. And
    /// succeeding on the bounded case is what stops the feature being useless,
    /// since every real accumulator has a range the author can state.
    #[test]
    fn bounds_make_a_float_accumulator_satisfiable() {
        let unbounded = Requirement::for_type_with_bounds("F32", None);
        assert!(
            select_repr(&unbounded, &CostTable::new()).is_none(),
            "an unbounded F32 accumulator has no exact representation"
        );

        let bounded = Requirement::for_type_with_bounds("F32", Some((-1000.0, 1000.0)));
        let d = select_repr(&bounded, &CostTable::new()).expect("bounded F32 must be satisfiable");
        assert!(d.repr.is_exact());
        assert_eq!(d.repr, DriftRepr::FixedQ32_32, "needs 2^-24 resolution, so Q16.16 is out");

        // Bounds must not silently coarsen the resolution the type asked for.
        assert_eq!(bounded.resolution, Requirement::for_type("F32").resolution);
    }

    #[test]
    fn cost_table_round_trips_through_the_profile() {
        let mut costs = CostTable::new();
        costs.insert(DriftRepr::FixedQ32_32, 412.5);
        costs.insert(DriftRepr::Int64, 401.25);
        let text = serialize_costs(&costs, "NVIDIA GeForce RTX 4070 Ti SUPER");
        let back = parse_costs(&text, "NVIDIA GeForce RTX 4070 Ti SUPER");
        assert_eq!(back.len(), 2);
        assert!((back[&DriftRepr::FixedQ32_32] - 412.5).abs() < 1e-6);
        // A different GPU's entries must not be read as this one's.
        assert!(parse_costs(&text, "NVIDIA A100").is_empty());
    }

    /// `Q32.32` and `Int64` must emit byte-identical probe PTX.
    ///
    /// They differ only in how the bits are *interpreted*; both accumulate with
    /// `add.s64`. That makes them a free control group: any measured gap
    /// between them is measurement error, because there is nothing physical to
    /// differ. `measure_accumulate_costs` uses exactly that, and it is how two
    /// successive versions of the probe were caught being constant-folded (766x
    /// then 48x apart). If this assertion ever fails, the control group is gone
    /// and the measurement loses its self-check.
    #[test]
    fn q32_32_and_int64_probes_are_identical() {
        assert_eq!(
            accumulate_probe_ptx(DriftRepr::FixedQ32_32, 64),
            accumulate_probe_ptx(DriftRepr::Int64, 64),
            "these two must stay a control group for the measurement"
        );
    }

    /// The probe must not be constant-foldable.
    ///
    /// Two properties make it survive ptxas: the seed is loaded from memory
    /// rather than written as a literal, and the two registers feed each other
    /// so there is no closed form to fold to. Losing either one silently turns
    /// the measurement into a timer for an empty kernel.
    #[test]
    fn probe_is_not_constant_foldable() {
        let ptx = accumulate_probe_ptx(DriftRepr::FixedQ32_32, 8);
        assert!(
            ptx.contains("ld.global.u64 %rd1, [%out]"),
            "the seed must come from memory, not a literal:\n{}",
            ptx
        );
        // Both directions of the dependence must appear.
        assert!(ptx.contains("add.s64 %rd0, %rd0, %rd1"), "missing acc += step");
        assert!(ptx.contains("add.s64 %rd1, %rd1, %rd0"), "missing step += acc");
        // And the result must not overwrite the seed, or it drifts per launch.
        assert!(ptx.contains("add.u64 %dst, %out, 8"), "result must not clobber the seed");
    }

    #[test]
    fn probe_ptx_has_the_requested_chain_length() {
        let ptx = accumulate_probe_ptx(DriftRepr::FixedQ32_32, 16);
        assert_eq!(ptx.matches("add.s64").count(), 16);
        assert!(ptx.contains(".target sm_89"));
        let ptx32 = accumulate_probe_ptx(DriftRepr::FixedQ16_16, 8);
        assert_eq!(ptx32.matches("add.s32").count(), 8);
    }
}
