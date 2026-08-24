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
    ///
    /// **Zero means two different things here, and callers branch on it.**
    /// `llvm_emitter` and `ptx_emitter` both test `repr.frac_bits() == 0` to
    /// decide whether to emit the scale/unscale pair, so for `Int64` the zero
    /// is a correct instruction ("accumulate in plain integers"). For the two
    /// float representations it is meaningless -- they are not fixed point and
    /// have no fractional-bit count -- and the old `_ => 0` made them
    /// indistinguishable from `Int64`.
    ///
    /// That is latent rather than live: `select_repr` refuses any repr failing
    /// `is_exact`, so `Float64` and `KahanF32` never reach an emitter. They are
    /// constructed only so the rejection report can name them. Written out
    /// exhaustively so that adding a representation is a compile error here
    /// instead of silently inheriting an integer lowering.
    pub fn frac_bits(self) -> u32 {
        match self {
            DriftRepr::FixedQ16_16 => 16,
            DriftRepr::FixedQ32_32 => 32,
            // Exact and genuinely has no fractional part.
            DriftRepr::Int64 => 0,
            // Not fixed point at all. Reached only by the rejection report,
            // which prints this beside "not exact"; never by a lowering.
            DriftRepr::Float64 | DriftRepr::KahanF32 => 0,
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

    /// The LLVM IR type the accumulator is stored in.
    pub fn llvm_type(self) -> &'static str {
        match self {
            DriftRepr::FixedQ16_16 => "i32",
            DriftRepr::FixedQ32_32 | DriftRepr::Int64 => "i64",
            DriftRepr::Float64 => "double",
            DriftRepr::KahanF32 => "float",
        }
    }

    /// The multiplier taking a real value into this representation's integer
    /// domain. `1.0` for the pure-integer representations.
    pub fn scale(self) -> f64 {
        (2f64).powi(self.frac_bits() as i32)
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
                if digits.len() == 2 && digits[0] >= 1 {
                    // `Q<i>.<f>` counts the SIGN BIT in `i`, so the magnitude
                    // it can hold is 2^(i-1), not 2^i - which is exactly what
                    // `DriftRepr::FixedQ32_32` provides (±2^31 in 64 bits).
                    //
                    // Asking for 2^i made every Q declaration unsatisfiable BY
                    // ITS OWN REPRESENTATION, so `@ZeroDrift let acc: Q32.32`
                    // was refused with a message advising the user to "declare
                    // it as a Q format" - which they had. Off by one bit, and
                    // it made the whole Q surface dead: no test built a
                    // `Requirement` through this path, they all constructed the
                    // struct directly.
                    Requirement {
                        max_magnitude: 2f64.powi(digits[0] as i32 - 1),
                        resolution: Some(2f64.powi(-(digits[1] as i32))),
                    }
                } else {
                    Requirement { max_magnitude: 3.4028235e38, resolution: Some(2f64.powi(-24)) }
                }
            }
            // An unrecognised type name gets F32's requirement, and that is
            // DELIBERATE rather than a guess: no exact representation can hold
            // 3.4e38 at 2^-24, so an unknown type is REFUSED by `select_repr`
            // unless `@bounds` narrows it. That is the design rule's first
            // option -- over-approximate so the obligation becomes harder --
            // and it is why this arm is not a `smt_unmodellable`-style error.
            //
            // Pinned by `an_unknown_type_name_is_refused_not_guessed`, because
            // a fail-closed default that nothing checks is one edit away from
            // becoming a fail-open one.
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
/// The one shape of assignment that is an exact accumulation.
///
/// `acc += e` has a drift-aware arm in every backend that lowers `@ZeroDrift`.
/// `acc = acc + e` is the SAME statement and had no arm in either - it fell
/// through to the ordinary assignment path, where the LLVM backend read the
/// integer accumulator as a `double`, added in `double`, and then
/// `store double` into an `alloca i64` (pointers are untyped, so clang said
/// nothing), and the PTX backend read it as **f32**, added in f32, and wrote
/// back with an UNSIGNED `cvt.rzi.u64.f32` - under a comment reading
/// `accumulated exactly as I64`.
///
/// It lives here, not in either emitter, because "which assignments preserve
/// drift-freedom" is a rule about the language and not about a target. A
/// second copy is how the two backends came to disagree in the first place;
/// `tests/zero_drift_backend_agreement.rs` pins that they now do not.
///
/// Returns the operator and the term being accumulated, for
/// `acc = acc + e` / `acc = acc - e` only. Everything else - `acc = e`,
/// `acc = e + acc`, `acc = acc * e` - is `None`, and the caller must REFUSE
/// rather than lower it: silently accepting `acc = acc * e` would reintroduce
/// rounding into the accumulation, which is the whole thing the directive
/// promises not to do.
///
/// `acc = e + acc` is deliberately not accepted even though addition is
/// commutative. Recognising it means deciding, per operator, whether the
/// accumulator may appear on the right - true for `+`, false for `-` - and the
/// refusal already names `+=` as the repair, which is one character.
pub fn running_sum<'e>(
    target: &crate::ast::Expr,
    value: &'e crate::ast::Expr,
) -> Option<(crate::ast::BinaryOp, &'e crate::ast::Expr)> {
    use crate::ast::{BinaryOp, Expr};
    let name = match target {
        Expr::Ident(n, _) => n,
        _ => return None,
    };
    match value {
        Expr::BinaryOp { op, left, right, .. }
            if matches!(op, BinaryOp::Add | BinaryOp::Sub)
                && matches!(&**left, Expr::Ident(l, _) if l == name) =>
        {
            Some((op.clone(), &**right))
        }
        _ => None,
    }
}

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

// ── Exact VNNI accumulation ─────────────────────────────────
//
// See `docs/deterministic_inference.md`, milestone M0, and
// `docs/proof_carrying_kernels.md`'s Phase 0 measurement, which is what makes
// this scheme worth licensing at all: exact `vpdpwssd` + int64 flush measured
// 314.5 G MAC/s against f32 FMA's 166.9, i.e. **1.88x faster than float**,
// verified exact against a scalar reference. Exactness here costs range, not
// speed.

/// The exact `vpdpwssd` accumulation scheme: int16 operands, products
/// accumulated in int32, flushed into an int64 accumulator every
/// `flush_k_pairs` k-pairs so the narrow accumulator can never overflow.
///
/// The reference implementation this describes is
/// `tests/probes/vnni_kernels.c::micro_vnni_exact`, which is measured and
/// verified exact. This type is the *licensing* half: it answers whether a
/// given nest's operands are small enough that the scheme is sound, so that a
/// nest which would overflow is refused with a reason instead of silently
/// producing a wrong answer.
///
/// **Why the check has to exist before the kernel does.** An int32 accumulator
/// that overflows does not signal; it wraps. The result is a plausible number
/// that is simply wrong, from a kernel whose entire purpose is to be exact —
/// the exact failure shape `CLAUDE.md`'s design rule is written against. So the
/// obligation is discharged at compile time, or the representation is not
/// selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VnniExact {
    /// k-pairs accumulated in int32 between flushes into int64.
    pub flush_k_pairs: u32,
}

impl VnniExact {
    /// The interval the measured probe uses.
    ///
    /// **Powers of two only, and the reason changed.** The C probe
    /// (`tests/probes/vnni_kernels.c`) tests `(p & (T-1)) == T-1` inside its
    /// hot loop, which is only an "every T iterations" test when T is a power
    /// of two. Y's emitted kernel no longer works that way — the flush was
    /// lifted into an outer chunk loop, so any positive T would encode
    /// correctly. The restriction is kept as a **canonical form**, not a
    /// hardware or encoding necessity: [`Self::flush_interval_for`] returns
    /// powers of two, so allowing others would mean two shapes of interval in
    /// circulation and one of them never generated.
    pub const DEFAULT_FLUSH_K_PAIRS: u32 = 64;

    /// int16 operands, so this bounds the magnitude regardless of the flush
    /// interval. `i16::MIN` is not representable as a positive magnitude, so
    /// the usable bound is `i16::MAX`.
    pub const OPERAND_WIDTH_LIMIT: f64 = 32767.0;

    pub fn new(flush_k_pairs: u32) -> Option<Self> {
        if flush_k_pairs == 0 || !flush_k_pairs.is_power_of_two() {
            return None;
        }
        Some(VnniExact { flush_k_pairs })
    }

    /// Largest operand magnitude for which the int32 accumulator provably
    /// cannot overflow between two flushes.
    ///
    /// Derivation, and it is worth writing out because every term matters:
    ///
    /// - `vpdpwssd` computes, per int32 lane,
    ///   `acc += a[2i]*b[2i] + a[2i+1]*b[2i+1]` — **two** MACs per lane per
    ///   instruction, not one.
    /// - The accumulator is zeroed at every flush, so the worst case is one
    ///   full flush interval starting from zero: `2 * flush_k_pairs` products.
    /// - With both operands bounded by `m`, each product is at most `m^2`.
    /// - Signed int32 holds up to `2^31 - 1`.
    ///
    /// So the obligation is `2 * flush_k_pairs * m^2 <= 2^31 - 1`, giving
    /// `m <= sqrt((2^31 - 1) / (2 * flush_k_pairs))`, and the int16 operand
    /// width caps it independently.
    ///
    /// At the default 64 k-pairs this yields **4095**, not the 1024 quoted in
    /// the probe's header comment. 1024 is sound - it is 4x inside this bound -
    /// but it is not the limit, and a licensing check that refuses a legal nest
    /// is a bug in the direction that merely costs performance rather than
    /// correctness. The derived figure is used, and the discrepancy is recorded
    /// here rather than silently reconciled.
    ///
    /// **The int16 operand width does not need to be applied here, and a
    /// `.min(OPERAND_WIDTH_LIMIT)` that was originally written into this
    /// function was dead code.** `flush_k_pairs >= 1`, so `products >= 2`, so
    /// the derivation is at most `sqrt((2^31 - 1) / 2) = 32767.99...`, which
    /// floors to exactly `i16::MAX`. The overflow bound is therefore always at
    /// least as tight as the width bound, and clamping to the width can never
    /// change the answer. Mutation testing is what established this - removing
    /// the clamp passed every test, because no reachable interval distinguishes
    /// the two. `the_derivation_can_never_exceed_int16` pins the invariant that
    /// makes the omission safe; [`Self::license`] still checks the width
    /// separately, because a *caller-supplied* magnitude is unconstrained and
    /// deserves the width-specific message.
    pub fn max_operand_magnitude(&self) -> f64 {
        let products = 2.0 * f64::from(self.flush_k_pairs);
        (f64::from(i32::MAX) / products).sqrt().floor()
    }

    /// Whether operands bounded by `m` in magnitude are licensed.
    ///
    /// `m` is in the scheme's own integer domain - the int16 operand values
    /// actually fed to `vpdpwssd` - **not** the real-valued range of the source
    /// matrices. Converting one to the other is the quantization scale, and it
    /// is the caller's job precisely because getting it wrong is a wrong answer
    /// rather than a slow one.
    pub fn licenses(&self, m: f64) -> bool {
        m.is_finite() && m >= 0.0 && m <= self.max_operand_magnitude()
    }

    /// Licence the scheme for operands bounded by `m`, or say why not.
    pub fn license(&self, m: f64) -> Result<(), String> {
        // The licence must be about the values the KERNEL sees, and `m` arrives
        // from `@bounds` on the source matrices - real numbers. Those coincide
        // only when staging into the int16 domain is the identity.
        //
        // A magnitude below 1 is the case where they provably do not: every
        // operand rounds to 0 or +-1, so the kernel computes something
        // unrelated to the source while this function certifies it exact.
        // `@bounds(-0.001, 0.001)` was licensed at magnitude 0.001, and the
        // int16 value of every such operand is zero.
        //
        // This is the same "necessary but not sufficient" shape `OperandBounds`
        // records one level in: there, an accumulator bound does not imply an
        // operand bound; here, a real-valued operand bound does not imply an
        // integer-domain one. Refused rather than guessed at a scale - the
        // design rule's whole subject.
        //
        // Zero is admitted deliberately: an all-zero operand IS representable,
        // and refusing it would refuse a degenerate but correct program.
        if m.is_finite() && m > 0.0 && m < 1.0 {
            return Err(format!(
                "operands bounded by {m} are not representable as int16: `vpdpwssd` consumes \
                 integers, so every operand of magnitude below 1 stages to 0 and the kernel would \
                 compute a different matrix than the source. State the bound in the scheme's \
                 integer domain - i.e. apply the quantization scale before licensing - or use the \
                 scalar exact lowering, which is exact and slow"
            ));
        }
        if !m.is_finite() || m < 0.0 {
            return Err(format!(
                "operand bound {m} is not a usable magnitude: exact VNNI accumulation needs a \
                 finite non-negative bound on both operands"
            ));
        }
        if m > Self::OPERAND_WIDTH_LIMIT {
            return Err(format!(
                "operands bounded by {m} do not fit int16 (limit {}): `vpdpwssd` reads its \
                 operands as pairs of int16, so a wider operand is a different instruction, not \
                 a wider accumulator",
                Self::OPERAND_WIDTH_LIMIT
            ));
        }
        let limit = self.max_operand_magnitude();
        if m > limit {
            return Err(format!(
                "operands bounded by {m} can overflow the int32 accumulator: {} k-pairs between \
                 flushes is {} products of up to {m}^2, which exceeds i32::MAX. Reduce the flush \
                 interval to at most {}, or state a tighter `@bounds` on the operands (limit at \
                 this interval is {limit})",
                self.flush_k_pairs,
                2 * self.flush_k_pairs,
                self.flush_interval_for(m),
            ));
        }
        Ok(())
    }

    /// Largest power-of-two flush interval that licenses operands bounded by
    /// `m`. Zero when no interval does - i.e. when a single product already
    /// overflows, which int16 operands cannot actually reach, but the caller
    /// should not have to know that to read the number.
    pub fn flush_interval_for(&self, m: f64) -> u32 {
        if !m.is_finite() || m <= 0.0 {
            return Self::DEFAULT_FLUSH_K_PAIRS;
        }
        let max_products = f64::from(i32::MAX) / (m * m);
        let max_pairs = (max_products / 2.0).floor();
        if max_pairs < 1.0 {
            return 0;
        }
        // Round down to a power of two: the flush test is a bitmask.
        let p = max_pairs.min(f64::from(u32::MAX)) as u32;
        1u32 << (31 - p.leading_zeros())
    }
}

impl Default for VnniExact {
    fn default() -> Self {
        VnniExact { flush_k_pairs: Self::DEFAULT_FLUSH_K_PAIRS }
    }
}

/// What a `@ZeroDrift` GEMM must state before the exact VNNI kernel can be
/// selected for it.
///
/// **The accumulator's `@bounds` is necessary but NOT sufficient, and this is
/// the finding that shapes M0.** `@bounds(min, max)` on the accumulator states
/// the range of the *result*, `C[i,j]`. What licenses the representation is the
/// range of the *operands*, `A[i,k]` and `B[k,j]` — and a bound on the sum
/// implies nothing about a bound on the terms, because the terms can cancel.
/// A result bounded by 1.0 is perfectly consistent with operands of 1e9.
///
/// So a nest carrying only an accumulator bound cannot be licensed, and saying
/// so is the whole value of this type: the alternative is to guess an operand
/// range, which is the "silent approximation" the design rule forbids.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OperandBounds {
    /// Magnitude bound on both operands, in the scheme's integer domain.
    pub max_magnitude: f64,
}

/// Decide whether the exact VNNI kernel may be emitted for a nest.
///
/// Returns the licensed scheme, or the reason it cannot be. `operands` being
/// `None` is itself a refusal rather than a default, for the reason in
/// [`OperandBounds`].
pub fn license_vnni_exact(
    operands: Option<OperandBounds>,
    flush_k_pairs: u32,
) -> Result<VnniExact, String> {
    let Some(scheme) = VnniExact::new(flush_k_pairs) else {
        return Err(format!(
            "flush interval {flush_k_pairs} is not a positive power of two: the flush test is a \
             bitmask, so a non-power-of-two interval would flush at the wrong iterations"
        ));
    };
    let Some(b) = operands else {
        return Err(
            "no operand bounds: `@bounds` on the accumulator states the range of the RESULT, and \
             a bound on a sum implies no bound on its terms - they can cancel. Exact VNNI \
             accumulation needs a stated magnitude bound on A and B themselves."
                .to_string(),
        );
    };
    scheme.license(b.max_magnitude)?;
    Ok(scheme)
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
        // sm_80, NOT sm_89 - this probe uses nothing newer than Ampere, and a
        // `.target` above the device is a hard load failure. It is JIT'd for
        // whatever card is present, so the measurement is still that card's.
        ".version 7.0\n.target sm_80\n.address_size 64\n\
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
    /// The `_ =>` in `Requirement::for_type` is a fail-closed default, and
    /// nothing checked that it stayed one.
    ///
    /// An unrecognised type name inherits F32's requirement. That is only safe
    /// because no exact representation can hold 3.4e38 at 2^-24, so the unknown
    /// type is refused rather than accumulated in something too narrow. Both
    /// halves are asserted: that F32 itself is unsatisfiable (which is the
    /// property doing the work), and that an unknown name behaves identically.
    ///
    /// If someone ever narrows that default -- to F16's range, say -- this
    /// fails, instead of the compiler quietly accepting a `@ZeroDrift`
    /// accumulator over a type it could not identify.
    #[test]
    fn an_unknown_type_name_is_refused_not_guessed() {
        let costs = CostTable::new();
        assert!(
            select_repr(&Requirement::for_type("F32"), &costs).is_none(),
            "F32's requirement must be unsatisfiable -- the unknown-type \
             default leans on exactly that"
        );
        for name in ["Widget", "", "Q", "Qx.y", "f80", "I128"] {
            assert!(
                select_repr(&Requirement::for_type(name), &costs).is_none(),
                "an unrecognised type name ({name:?}) must be refused, not \
                 given a representation chosen for a type nobody identified"
            );
        }
        // The control: the default must not be refusing EVERYTHING, or the
        // assertions above would hold with `for_type` returning garbage.
        assert!(
            select_repr(&Requirement::for_type("I32"), &costs).is_some(),
            "a type that IS representable must still be accepted"
        );
        // And `@bounds` is the documented way to make a wide type work.
        assert!(
            select_repr(
                &Requirement::for_type_with_bounds("F32", Some((-1000.0, 1000.0))),
                &costs
            )
            .is_some(),
            "@bounds must rescue a declared F32 by stating its real range"
        );
    }

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
        // sm_80, deliberately: see `accumulate_probe_ptx`. Targeting the
        // local card's arch would make the probe unloadable on any older one.
        assert!(ptx.contains(".target sm_80"));
        let ptx32 = accumulate_probe_ptx(DriftRepr::FixedQ16_16, 8);
        assert_eq!(ptx32.matches("add.s32").count(), 8);
    }

    // ── Exact VNNI licensing ────────────────────────────────

    /// The bound is DERIVED from the flush interval, not hardcoded.
    ///
    /// This is the arithmetic the whole scheme rests on, so it is asserted
    /// against a hand-computed value rather than against the implementation:
    /// at 64 k-pairs there are 128 products, and `sqrt((2^31 - 1) / 128)` is
    /// 4095.999..., which floors to 4095.
    #[test]
    fn the_operand_bound_is_derived_from_the_flush_interval() {
        let v = VnniExact::default();
        assert_eq!(v.flush_k_pairs, 64);
        assert_eq!(v.max_operand_magnitude(), 4095.0);

        // Halving the interval doubles the product budget, so the magnitude
        // bound grows by sqrt(2).
        let half = VnniExact::new(32).unwrap();
        assert_eq!(half.max_operand_magnitude(), 5792.0);

        // And a long enough interval drives it below the probe's stated 1024.
        let long = VnniExact::new(1024).unwrap();
        assert!(
            long.max_operand_magnitude() < 1024.0,
            "1024 k-pairs must not license operands of 1024: {}",
            long.max_operand_magnitude()
        );
    }

    /// **The overflow bound is always at least as tight as the int16 width
    /// bound, at every legal flush interval.**
    ///
    /// This is the invariant that lets [`VnniExact::max_operand_magnitude`]
    /// omit a width clamp. It is not obvious and it was not asserted at first:
    /// mutation testing removed the clamp and every test still passed, because
    /// the smallest legal interval makes the two bounds *exactly equal* rather
    /// than making one redundant by a margin. `products = 2 * flush_k_pairs`,
    /// so the loosest possible derivation is `sqrt((2^31 - 1) / 2)`, which
    /// floors to 32767 = `i16::MAX`.
    ///
    /// If the scheme ever changes so a `vpdpwssd` lane carries a different
    /// number of MACs, this test fails and the clamp has to come back.
    #[test]
    fn the_derivation_can_never_exceed_int16() {
        // The boundary case: one k-pair, the loosest interval that exists.
        let loosest = VnniExact::new(1).unwrap();
        assert_eq!(
            loosest.max_operand_magnitude(),
            VnniExact::OPERAND_WIDTH_LIMIT,
            "the smallest interval must land exactly on i16::MAX, not above it"
        );

        // And every other legal interval is tighter still.
        for shift in 0..20 {
            let v = VnniExact::new(1u32 << shift).unwrap();
            assert!(
                v.max_operand_magnitude() <= VnniExact::OPERAND_WIDTH_LIMIT,
                "interval {} derives {}, above int16",
                v.flush_k_pairs,
                v.max_operand_magnitude()
            );
        }
    }

    /// A caller-supplied magnitude above int16 gets the WIDTH message, not the
    /// overflow one.
    ///
    /// This is the reachable half of the width check. `max_operand_magnitude`
    /// never produces such a value, but nothing stops a caller passing one, and
    /// "does not fit int16" points at a different fix than "reduce the flush
    /// interval" - the latter cannot help at any interval.
    #[test]
    fn an_operand_wider_than_int16_is_refused_for_the_right_reason() {
        let v = VnniExact::default();
        let err = v.license(40000.0).expect_err("40000 does not fit int16");
        assert!(
            err.contains("int16"),
            "must name the width, not just the overflow: {err}"
        );
        assert!(
            !err.contains("Reduce the flush interval"),
            "a narrower interval cannot make 40000 fit int16, so it must not be suggested: {err}"
        );
    }

    /// A nest that would overflow is REFUSED, and the refusal names the fix.
    #[test]
    fn an_overflowing_nest_is_refused_with_a_workable_reason() {
        let v = VnniExact::default();
        assert!(v.licenses(4095.0));
        assert!(!v.licenses(4096.0));

        let err = v.license(8192.0).expect_err("8192 must not be licensed at 64 k-pairs");
        assert!(err.contains("overflow"), "must say what goes wrong: {err}");
        // The suggested interval must actually license the operand.
        let suggested = v.flush_interval_for(8192.0);
        assert!(suggested > 0 && suggested.is_power_of_two(), "got {suggested}");
        let fixed = VnniExact::new(suggested).unwrap();
        assert!(
            fixed.licenses(8192.0),
            "the interval the error suggests ({suggested}) must license the operand it was \
             computed for, or the message sends the user in a circle"
        );
    }

    /// A non-power-of-two flush interval is refused, not rounded.
    ///
    /// The probe flushes on `(p & (FLUSH_T - 1)) == FLUSH_T - 1`. At
    /// `FLUSH_T = 48` that mask is `0b101111`, which fires on a set of
    /// iterations that is not "every 48th" - so the accumulator would run
    /// longer than the licence assumed. Silently rounding would be a licence
    /// computed for one schedule applied to another.
    #[test]
    fn a_non_power_of_two_flush_interval_is_refused() {
        assert!(VnniExact::new(48).is_none());
        assert!(VnniExact::new(0).is_none());
        assert!(VnniExact::new(64).is_some());

        let b = Some(OperandBounds { max_magnitude: 100.0 });
        let err = license_vnni_exact(b, 48).expect_err("48 must be refused");
        assert!(err.contains("power of two"), "must name the reason: {err}");
    }

    /// **The finding that shapes M0: missing operand bounds are a REFUSAL.**
    ///
    /// `@bounds` on the accumulator constrains the result. The licence needs
    /// the operands, and a bound on a sum implies nothing about its terms
    /// because they cancel - a result bounded by 1.0 is consistent with
    /// operands of 1e9. Defaulting to any operand range here would be exactly
    /// the silent approximation the design rule forbids, so it must refuse.
    #[test]
    fn missing_operand_bounds_are_a_refusal_not_a_default() {
        let err = license_vnni_exact(None, VnniExact::DEFAULT_FLUSH_K_PAIRS)
            .expect_err("no operand bounds must refuse");
        assert!(
            err.contains("cancel"),
            "the refusal must explain why the accumulator's own bound does not transfer: {err}"
        );

        // With bounds, the same call succeeds - so the refusal is about the
        // missing information, not a path that never licenses anything.
        let ok = license_vnni_exact(
            Some(OperandBounds { max_magnitude: 1024.0 }),
            VnniExact::DEFAULT_FLUSH_K_PAIRS,
        )
        .expect("bounded operands must be licensed");
        assert_eq!(ok.flush_k_pairs, 64);
    }

    /// The probe's own stated bound must be licensed by this checker.
    ///
    /// `tests/probes/vnni_kernels.c` is measured, exact-verified, and states
    /// `|a|,|b| <= 1024` at `FLUSH_T = 64`. If the licensing logic refused the
    /// configuration that was actually validated on hardware, the logic would
    /// be wrong rather than the probe.
    #[test]
    fn the_measured_probe_configuration_is_licensed() {
        let v = VnniExact::new(64).expect("the probe's FLUSH_T");
        assert!(
            v.licenses(1024.0),
            "the configuration measured at 1.88x must be licensed by its own checker"
        );
    }
}
