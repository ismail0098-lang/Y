// ============================================================
//  Y — ZK Circuit Backend Emitter
//  zk_emitter.rs
//
//  Translates Y AST / IR into Rank-1 Constraint Systems (R1CS)
//  of the form (A · x) * (B · x) = C · x over the BN254 Fr field.
// ============================================================

#![allow(dead_code)]

use crate::ast::*;
use crate::zk_poseidon_constants::{
    POSEIDON_C_T3, POSEIDON_M_T3, POSEIDON_P_T3, POSEIDON_S_T3, POSEIDON_T3_ROUNDS_F,
    POSEIDON_T3_ROUNDS_P,
};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::rc::Rc;

// ────────────────────────────────────────────────────────
// 1. Field arithmetic
// ────────────────────────────────────────────────────────
//
// `BigUint` and `Fr` live in `zk_field.rs`. `Fr` is a `Copy` `[u64; 4]` in
// Montgomery form, not a heap `BigUint` - see that file's header for why, and
// `docs/zk_emit_profile.md` for the measurement that forced it. Re-exported
// here because `y::zk_emitter::{BigUint, Fr}` is the path every caller and test
// already uses.

pub use crate::zk_field::{active_modulus, field_op_counts, set_active_modulus, BigUint, Fr};


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarField {
    Bn254,
    Bls12_381,
    Pallas,
    Vesta,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofScheme {
    R1cs,
    Plonkish,
}

#[derive(Clone, Debug)]
pub struct FieldConfig {
    pub name: String,
    pub p: BigUint,
    pub capacity_bits: usize,
    pub capacity_bytes: usize,
    pub mds_matrix: Vec<Vec<Fr>>,
    pub round_constants: Vec<Fr>,
}

impl FieldConfig {
    pub fn get(field: ScalarField) -> Self {
        let (name, p_str, cap_bits, cap_bytes) = match field {
            ScalarField::Bn254 => (
                "bn254",
                "21888242871839275222246405745257275088548364400416034343698204186575808495617",
                253,
                31,
            ),
            ScalarField::Bls12_381 => (
                "bls12_381",
                "52435875175126190479447740508185965837690552500527637822603658699938581184513",
                254,
                31,
            ),
            ScalarField::Pallas => (
                "pallas",
                "28948022309329048855892746252171976963363056481941560715954676764349967630337",
                254,
                31,
            ),
            // Vesta's base field, `q` in the Pasta parameters, matching the
            // Pallas entry above (its base field `p`).
            //
            // This constant was WRONG until measured: it read
            // ...941600134020817490249052636161, which is COMPOSITE - a
            // corrupted transcription sharing only the leading
            // `0x40000000000000000000000000000000224698fc09` with the real
            // value. Circuits emitted against it were over a ring, not a
            // field. `FieldParams::new` now refuses a composite modulus, and
            // `all_supported_moduli_are_montgomery_ready` pins all four.
            ScalarField::Vesta => (
                "vesta",
                "28948022309329048855892746252171976963363056481941647379679742748393362948097",
                254,
                31,
            ),
        };
        let p = BigUint::from_str(p_str);

        // Temporarily set active modulus so elements are reduced properly
        let prev_modulus = active_modulus();
        set_active_modulus(&p);

        // Generate deterministic MDS matrix (Cauchy matrix)
        let mut mds = vec![vec![Fr::zero(); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let val = (i + j + 5) as u64;
                mds[i][j] = Fr::from_u64(val).inv();
            }
        }

        // Generate deterministic round constants
        let mut rc = Vec::new();
        let mut seed = match field {
            ScalarField::Bn254 => 0x12345678u64,
            ScalarField::Bls12_381 => 0x87654321u64,
            ScalarField::Pallas => 0xabcdef01u64,
            ScalarField::Vesta => 0x10fedcbau64,
        };
        for _ in 0..60 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            rc.push(Fr::from_u64(seed));
        }

        // Restore previous modulus
        set_active_modulus(&prev_modulus);

        FieldConfig {
            name: name.to_string(),
            p,
            capacity_bits: cap_bits,
            capacity_bytes: cap_bytes,
            mds_matrix: mds,
            round_constants: rc,
        }
    }
}

// ────────────────────────────────────────────────────────
// 1.5. Witness IR Structures for GPU PTX Witness Generator
// ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SignalId(pub usize);

#[derive(Clone, Debug)]
pub enum FieldType {
    Bn254,
    Goldilocks,
    BabyBear,
}

#[derive(Clone, Debug)]
pub enum HintOp {
    NonDeterministicInv { src: SignalId, dst: SignalId },
    BitDecompose { src: SignalId, dst_bits: Vec<SignalId> },
    AssignExpr { dst: SignalId, src: SignalId },
}

#[derive(Clone, Debug)]
pub enum WitnessOp {
    Const(Fr),
    LoadInput { input_idx: usize, is_public: bool },
    Add(SignalId, SignalId),
    Sub(SignalId, SignalId),
    Mul(SignalId, SignalId),
    Div(SignalId, SignalId),
    Inv(SignalId),
    AssertEq(SignalId, SignalId),
    HintBlock {
        inputs: Vec<SignalId>,
        outputs: Vec<SignalId>,
        ops: Vec<HintOp>,
    },

    // ---- linear-combination recipes ----
    //
    // The variants above all take `SignalId`s, which is enough for wires the
    // constraint scan in `build_witness_ir` can reconstruct (`a * b = c` with
    // one term per side). The gadgets below cannot be reconstructed that way:
    // an is-zero flag and its inverse witness appear together in constraints
    // that each have TWO unknowns, so no amount of algebraic back-propagation
    // pins either one. That is exactly why every circuit containing `==`, `!=`
    // or a comparison used to come back `satisfied = false` from
    // `solve_r1cs_witness` - unprovable, not merely slow.
    //
    // These carry the linear combination directly so the witness pass can
    // evaluate it against the wires it has already solved.
    /// 1 if the linear combination evaluates to zero, else 0.
    IsZeroLc(LinearCombination),
    /// The inverse of the linear combination, or 0 when it is zero (the
    /// standard non-deterministic advice for an is-zero gadget).
    InvOrZeroLc(LinearCombination),
    /// Bit `bit` of the linear combination's value, LSB-first.
    BitOfLc { lc: LinearCombination, bit: u32 },
    /// Integer quotient `floor(a / b)`, or 0 when `b` is zero.
    ///
    /// Integer division is not field division and cannot be recovered from the
    /// constraint `q * b = a - r`, which has two unknowns. The zero-divisor
    /// case returns 0 only so the forward pass terminates; the accompanying
    /// `r < b` check is unsatisfiable when `b = 0`, so no proof follows from it.
    IntDivLc(LinearCombination, LinearCombination),
    /// Integer remainder `a mod b`, or 0 when `b` is zero. See `IntDivLc`.
    IntModLc(LinearCombination, LinearCombination),
    /// `a * b + c`, all three linear combinations.
    ///
    /// R1CS lets one constraint carry `A * B = C` with arbitrary linear
    /// combinations on every side, so `out <== a*b + c` is ONE constraint
    /// (`a * b = out - c`). Reconstructing `out` from that constraint alone is
    /// impossible - it has two unknowns - so the recipe carries the fused form.
    /// Without it the circom front end would have to spend a second constraint
    /// materialising `a*b` into its own wire, which is exactly the kind of
    /// gratuitous size difference that makes a drop-in replacement not one.
    MulAddLc(LinearCombination, LinearCombination, LinearCombination),
    /// Field division `a * b^-1`, or 0 when `b` is 0.
    ///
    /// Only ever reached from circom's `<--`, which assigns a witness value and
    /// deliberately emits NO constraint. That is the operator's whole purpose
    /// and also its danger; see `AssignOp::SignalOnly`.
    DivLc(LinearCombination, LinearCombination),
    /// `if lc == 0 { then } else { else_ }`, evaluated at witness time.
    ///
    /// circom's ternary over signals, which is legal in a `<--` and nowhere
    /// else. It exists because circomlib's `IsZero` is
    /// `inv <-- in != 0 ? 1/in : 0` - and `IsZero` is underneath `IsEqual`,
    /// `ForceEqualIfEnabled`, the SMT circuits, `Multiplexer` and every EdDSA
    /// verifier, so without it a third of circomlib does not compile.
    ///
    /// Both comparisons reduce to this one: `a == b ? t : e` is
    /// `IfZeroLc(a - b, t, e)` and `a != b ? t : e` swaps the branches.
    ///
    /// Only the taken branch is evaluated. In `IsZero` the untaken branch at
    /// `in = 0` is `1/in`, and `DivLc` happens to return 0 for a zero divisor
    /// rather than trapping - so eager evaluation would agree here, today, by
    /// coincidence of that convention. Mutation-checked: removing the laziness
    /// breaks nothing currently. It is kept so that correctness does not rest on
    /// a neighbouring variant's choice about division by zero, and so a future
    /// branch op that traps or is costly does not have to rediscover this.
    IfZeroLc(LinearCombination, Box<WitnessOp>, Box<WitnessOp>),
    /// The product of two linear combinations.
    ///
    /// R1CS lets the `A` and `B` of a constraint be arbitrary linear
    /// combinations, and Poseidon's S-box uses that: `x2 = (state + C) * (state
    /// + C)` is one constraint with no wire allocated for `state + C`. But the
    /// `a * b = c` scan in `build_witness_ir` only recognises constraints with
    /// a SINGLE term per side, so it cannot reconstruct those outputs and would
    /// hand all 243 of a Poseidon's wires to back-propagation - correct, but it
    /// drops the circuit off the fast path that made witness generation linear.
    MulLc(LinearCombination, LinearCombination),
    /// This wire's value cannot be reconstructed by the forward pass; leave it
    /// for the R1CS back-propagation in `solve_r1cs_witness`.
    ///
    /// The fallback here used to be `Const(0)`, which was actively harmful:
    /// `solve_r1cs_witness` treats `Const` as already-solved, so any wire the
    /// constraint scan could not recognise was pinned to ZERO and
    /// back-propagation was never allowed to correct it. That silently broke
    /// every output computed as `constant - wire` (e.g. `1 - lt`, `1 - eq`),
    /// which is exactly the shape every comparison and `!=` produces.
    Unknown,
}

#[derive(Clone, Debug)]
pub struct WitnessIRGraph {
    pub field: FieldType,
    pub num_public_inputs: usize,
    pub num_private_inputs: usize,
    pub num_signals: usize,
    pub nodes: Vec<WitnessOp>,
    pub signal_names: HashMap<usize, String>,
    pub topological_order: Vec<SignalId>,
}

// ────────────────────────────────────────────────────────
// 2. R1CS Structural Definitions & Linear Combinations
// ────────────────────────────────────────────────────────

thread_local! {
    /// Calls to `simplify` that actually scanned, and the total terms they
    /// scanned. **Always compiled in, because this is the number that regresses
    /// silently**: `simplify`'s cost is invisible in a profile split by phase
    /// (it is spread across all of `emit_circuit_entry`) and invisible in the
    /// field-op counters (it does no field arithmetic on its fast path). The
    /// dot-product circuit was O(N²) for exactly that reason.
    ///
    /// A thread-local `Cell` rather than an atomic: the ZK emitter is
    /// single-threaded, and this sits on its hottest path.
    static LC_SIMPLIFY_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static LC_SIMPLIFY_TERMS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// `(calls, terms_scanned)` since `reset_lc_simplify_stats`.
///
/// `terms_scanned` is the one to watch: it must stay **linear** in circuit size.
/// If it grows quadratically, some caller is re-establishing the sorted-and-
/// distinct invariant on a linear combination that never lost it — see
/// `LinearCombination::append_keeps_order`.
pub fn lc_simplify_stats() -> (u64, u64) {
    (LC_SIMPLIFY_CALLS.with(|c| c.get()), LC_SIMPLIFY_TERMS.with(|c| c.get()))
}

pub fn reset_lc_simplify_stats() {
    LC_SIMPLIFY_CALLS.with(|c| c.set(0));
    LC_SIMPLIFY_TERMS.with(|c| c.set(0));
}

#[derive(Clone, Debug)]
pub struct LinearCombination {
    // List of (wire_id, coefficient). By convention, wire_id = 0 is constant 1
    pub terms: Vec<(usize, Fr)>,
    pub is_simplified: bool,
}

impl LinearCombination {
    pub fn zero() -> Self {
        Self { terms: Vec::new(), is_simplified: true }
    }

    pub fn constant(val: Fr) -> Self {
        if val.is_zero() {
            Self::zero()
        } else {
            Self { terms: vec![(0, val)], is_simplified: true }
        }
    }

    pub fn variable(id: usize) -> Self {
        Self { terms: vec![(id, Fr::one())], is_simplified: true }
    }

    /// Would appending a block of terms whose smallest wire id is `first` keep
    /// the `is_simplified` invariant — strictly ascending wire ids, no zero
    /// coefficients?
    ///
    /// **This is the whole reason accumulating into a linear combination is
    /// linear rather than quadratic.** `simplify` re-establishes the invariant
    /// by scanning every term, which costs O(len) even on its "already sorted"
    /// fast path. An accumulator (`sum = sum + a * b` in a loop) grows by one
    /// term per iteration and was simplified once per iteration, so the emitter
    /// scanned N²/2 terms to re-derive a property that appending a fresh,
    /// larger wire id had never broken: measured at N=40,000, 119,999 calls but
    /// **800,259,997 terms**. Deciding it here, from the boundary alone, is O(1)
    /// and `simplify` then returns immediately.
    #[inline]
    fn append_keeps_order(&self, first: usize) -> bool {
        match self.terms.last() {
            // An empty combination trivially satisfies the invariant, whatever
            // the flag happens to say.
            None => true,
            Some((last, _)) => self.is_simplified && *last < first,
        }
    }

    pub fn add_constant(&mut self, val: Fr) {
        if !val.is_zero() {
            // Wire 0 sorts before everything, so this only stays ordered when
            // there is nothing to sort against.
            let keep = self.append_keeps_order(0);
            self.terms.push((0, val));
            self.is_simplified = keep;
        }
    }

    pub fn add_term(&mut self, wire_id: usize, val: Fr) {
        if val.is_zero() {
            return;
        }
        if self.append_keeps_order(wire_id) {
            self.terms.push((wire_id, val));
            self.is_simplified = true;
            return;
        }
        if self.is_simplified {
            if let Ok(idx) = self.terms.binary_search_by_key(&wire_id, |t| t.0) {
                let merged = self.terms[idx].1.add(&val);
                self.terms[idx].1 = merged;
                // A coefficient that cancelled to zero breaks the other half of
                // the invariant; leave it for `simplify` to filter out.
                self.is_simplified = !merged.is_zero();
                return;
            }
        }
        self.terms.push((wire_id, val));
        self.is_simplified = false;
    }

    /// Fold `other * scale` into `self` by updating coefficients in place, for
    /// the case where every wire of `other` **already has a slot** in `self`.
    ///
    /// This is the second half of keeping accumulation linear, and it covers a
    /// shape `append_keeps_order` cannot: `s = s + a * y; s = s + x;` in a loop.
    /// The first statement appends a freshly allocated (largest) wire and takes
    /// the append shortcut; the second adds `x`, an *input* wire with a small
    /// id, which is out of order and used to force a full sort of the whole
    /// accumulator — putting the circuit straight back to O(N²) (measured
    /// 2,010,998 → 8,021,998 → 32,043,998 terms for N = 2,000 → 4,000 → 8,000).
    ///
    /// But an out-of-order wire is necessarily an *older* one, and after its
    /// first appearance it is already present, so there is nothing to reorder —
    /// only a coefficient to add. Two binary-search passes rather than one plus
    /// a scratch vector, so that a missing wire leaves `self` untouched without
    /// allocating: `other` is one or two terms wide in practice.
    fn merge_in_place(&mut self, other: &Self, scale: Fr) -> bool {
        if !self.is_simplified || !other.is_simplified {
            return false;
        }
        for (w, _) in &other.terms {
            if self.terms.binary_search_by_key(w, |t| t.0).is_err() {
                return false;
            }
        }
        let mut cancelled = false;
        for (w, coeff) in &other.terms {
            let idx = self
                .terms
                .binary_search_by_key(w, |t| t.0)
                .expect("checked present above");
            let merged = self.terms[idx].1.add(&coeff.mul(&scale));
            self.terms[idx].1 = merged;
            cancelled |= merged.is_zero();
        }
        if cancelled {
            self.is_simplified = false;
        }
        true
    }

    pub fn add_linear(&mut self, other: &Self, scale: Fr) {
        if scale.is_zero() || other.terms.is_empty() {
            return;
        }
        // `other.is_simplified` is what makes `other.terms[0].0` the smallest of
        // the block and guarantees none of its coefficients is zero; `scale` is
        // non-zero and Fr is a field, so the scaled coefficients are non-zero
        // too. Short-circuits, so the index is only reached once that holds.
        if other.is_simplified && self.append_keeps_order(other.terms[0].0) {
            for (wire, coeff) in &other.terms {
                self.terms.push((*wire, coeff.mul(&scale)));
            }
            self.is_simplified = true;
            return;
        }
        if self.merge_in_place(other, scale) {
            return;
        }
        for (wire, coeff) in &other.terms {
            self.terms.push((*wire, coeff.mul(&scale)));
        }
        self.is_simplified = false;
    }

    /// The invariant `is_simplified` claims: strictly ascending wire ids and no
    /// zero coefficients. For tests and debugging only — this is O(n), and the
    /// entire point of the flag is not to pay that.
    pub fn invariant_holds(&self) -> bool {
        if !self.is_simplified {
            return true; // claims nothing
        }
        self.terms.windows(2).all(|w| w[0].0 < w[1].0)
            && self.terms.iter().all(|(_, c)| !c.is_zero())
    }

    /// Scale in place. `scale` allocates a fresh `Vec`; the circom front end
    /// scales values it already owns and immediately drops the original.
    pub fn scale_assign(&mut self, factor: Fr) {
        if factor.is_zero() {
            self.terms.clear();
            self.is_simplified = true;
            return;
        }
        for (_, c) in self.terms.iter_mut() {
            *c = c.mul(&factor);
        }
        // Ordering is untouched and a non-zero factor cannot zero a coefficient
        // in a field, so the invariant survives verbatim.
    }

    pub fn scale(&self, factor: Fr) -> Self {
        if factor.is_zero() {
            return Self::zero();
        }
        let terms = self.terms.iter().map(|(w, c)| (*w, c.mul(&factor))).collect();
        Self { terms, is_simplified: self.is_simplified }
    }

    pub fn simplify(&mut self) {
        if self.is_simplified {
            return;
        }
        LC_SIMPLIFY_CALLS.with(|c| c.set(c.get() + 1));
        LC_SIMPLIFY_TERMS.with(|c| c.set(c.get() + self.terms.len() as u64));
        if self.terms.len() <= 1 {
            if self.terms.len() == 1 && self.terms[0].1.is_zero() {
                self.terms.clear();
            }
            self.is_simplified = true;
            return;
        }

        // Check if terms are already sorted and have no duplicate wire IDs
        let mut already_simple = true;
        for i in 0..self.terms.len() - 1 {
            if self.terms[i].0 >= self.terms[i+1].0 {
                already_simple = false;
                break;
            }
        }
        if already_simple {
            let has_zeros = self.terms.iter().any(|(_, coeff)| coeff.is_zero());
            if !has_zeros {
                self.is_simplified = true;
                return; // Already sorted, distinct, and non-zero
            }
        }

        // Sort and merge in place. This was a `HashMap<usize, Fr>` built and
        // torn down per call, which is two allocations plus a hash per term for
        // a job that is a sort over a handful of `(u32, [u64;4])` pairs - and
        // the sort was there anyway, because the output has to come out ordered.
        // `substitute_linear_constraints` rebuilds a linear combination for
        // every term it rewrites, so this is on the hottest path in the ZK
        // front end.
        self.terms.sort_unstable_by_key(|t| t.0);
        let mut w = 0;
        for r in 0..self.terms.len() {
            if w > 0 && self.terms[w - 1].0 == self.terms[r].0 {
                let add = self.terms[r].1;
                self.terms[w - 1].1 = self.terms[w - 1].1.add(&add);
            } else {
                self.terms[w] = self.terms[r];
                w += 1;
            }
        }
        self.terms.truncate(w);
        self.terms.retain(|(_, coeff)| !coeff.is_zero());
        self.is_simplified = true;
    }

    pub fn is_constant(&self) -> Option<Fr> {
        if self.terms.is_empty() {
            return Some(Fr::zero());
        }
        
        // A linear combination is not a constant if it contains any variable wire (id > 0)
        let mut has_variables = false;
        for (wire, coeff) in &self.terms {
            if *wire != 0 && !coeff.is_zero() {
                has_variables = true;
                break;
            }
        }
        if has_variables {
            return None;
        }

        // It only contains constant terms (wire 0), sum them up
        let mut sum = Fr::zero();
        for (wire, coeff) in &self.terms {
            if *wire == 0 {
                sum = sum.add(coeff);
            }
        }
        Some(sum)
    }

    pub fn to_string(&self, var_names: &HashMap<usize, String>) -> String {
        let mut simplified = self.clone();
        simplified.simplify();
        if simplified.terms.is_empty() {
            return "0".to_string();
        }
        let mut s = String::new();
        for (i, (wire, coeff)) in simplified.terms.iter().enumerate() {
            if i > 0 {
                s.push_str(" + ");
            }
            let name = if *wire == 0 {
                "1".to_string()
            } else {
                var_names.get(wire).cloned().unwrap_or_else(|| format!("w_{}", wire))
            };
            if *coeff == Fr::one() {
                s.push_str(&name);
            } else {
                s.push_str(&format!("{} * {}", coeff.to_string(), name));
            }
        }
        s
    }
}

impl PartialEq for LinearCombination {
    fn eq(&self, other: &Self) -> bool {
        self.terms == other.terms
    }
}

impl Eq for LinearCombination {}

impl std::hash::Hash for LinearCombination {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.terms.hash(state);
    }
}

#[derive(Clone, Debug)]
pub struct Constraint {
    pub a: LinearCombination,
    pub b: LinearCombination,
    pub c: LinearCombination,
    pub span: Option<Span>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum WireBinding {
    Wire(usize),
    Linear(LinearCombination),
}

#[derive(Clone, Debug)]
pub struct Circuit {
    pub num_variables: usize,
    pub variables: Vec<String>,
    pub public_inputs: Vec<usize>,
    pub private_inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub constraints: Vec<Constraint>,
}

/// A borrowed circuit, for consumers that only read it.
///
/// `build_circuit` deep-clones every `Constraint`, and the constraint list is by
/// far the largest thing the emitter holds - on a dense circuit it is ~1 KB per
/// constraint of `LinearCombination` terms. Handing that clone to the `.r1cs`
/// writer doubled peak memory for the duration of the write, on the exact path
/// that is supposed to be Y's advantage at circuit sizes other compilers cannot
/// reach. Owning `Circuit` stays for callers that genuinely need it.
#[derive(Copy, Clone, Debug)]
pub struct CircuitView<'a> {
    pub num_variables: usize,
    pub variables: &'a [String],
    pub public_inputs: &'a [usize],
    pub private_inputs: &'a [usize],
    pub outputs: &'a [usize],
    pub constraints: &'a [Constraint],
}

impl Circuit {
    pub fn view(&self) -> CircuitView<'_> {
        CircuitView {
            num_variables: self.num_variables,
            variables: &self.variables,
            public_inputs: &self.public_inputs,
            private_inputs: &self.private_inputs,
            outputs: &self.outputs,
            constraints: &self.constraints,
        }
    }
}

// ────────────────────────────────────────────────────────
// 3. Lowering and Optimization Pass
// ────────────────────────────────────────────────────────

/// Bit width used for ordering comparisons (`<`, `<=`, `>`, `>=`).
///
/// Y's integer type is `I32`, so 32 bits is the natural range for both
/// operands. An operand outside `[0, 2^32)` makes its range check
/// unsatisfiable, which means no proof can be produced - fail-closed, never
/// silently wrong. See `emit_less_than`.
pub const ZK_COMPARISON_BITS: u32 = 32;

/// A linear combination's value as a `u64`, when it is a compile-time constant
/// that fits in one. Used to constant-fold the integer gadgets, all of which
/// are expensive enough that folding is worth a check.
fn lc_u64(lc: &LinearCombination) -> Option<u64> {
    lc.is_constant()?.to_u64()
}

/// A cheap 64-bit mixer for hash-table bucketing.
///
/// Not cryptographic, and does not need to be: every bucket hit in
/// `optimize_circuit` is confirmed by a full `LinearCombination` equality, so a
/// collision costs one comparison and never a wrong answer. The pass previously
/// used `DefaultHasher` (SipHash) over the derived `Hash`, which meant hashing
/// ~2.2 KB of term data per constraint through a keyed permutation - about half
/// a gigabyte, and a large share of the pass, to compute a bucket index.
#[inline(always)]
fn fx_mix(hash: u64, word: u64) -> u64 {
    const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
    (hash.rotate_left(5) ^ word).wrapping_mul(SEED)
}

fn hash_lc(lc: &LinearCombination) -> u64 {
    let mut h = fx_mix(0, lc.terms.len() as u64);
    for (wire, coeff) in &lc.terms {
        h = fx_mix(h, *wire as u64);
        for limb in coeff.mont_limbs() {
            h = fx_mix(h, limb);
        }
    }
    h
}

/// Point every term at its replacement wire, re-simplifying if anything moved.
///
/// `subst` is a DENSE map indexed by wire id, identity where nothing changed.
///
/// Simplification is required, not cosmetic: two distinct wires can collapse
/// onto the same one, and the resulting duplicate terms must be summed (they may
/// even cancel to zero).
fn replace_wires_in_lc(lc: &mut LinearCombination, subst: &[usize]) {
    let mut changed = false;
    for term in &mut lc.terms {
        if let Some(&new_w) = subst.get(term.0) {
            if new_w != term.0 {
                term.0 = new_w;
                changed = true;
            }
        }
    }
    if changed {
        lc.is_simplified = false;
        lc.simplify();
    }
}

// How many extra terms `substitute_linear_constraints` may create per wire it
// eliminates; `None` disables the pass.
//
// Thread-local rather than a global or a process-wide `env::set_var`, so that
// `zk_linear_substitution.rs` can compile the same circuit with and without the
// pass and compare the two without racing the rest of the suite.
// `Y_ZK_LINSUB_BUDGET=off|<n>` seeds it per thread.
thread_local! {
    static LINSUB_BUDGET: std::cell::Cell<Option<usize>> =
        std::cell::Cell::new(linsub_budget_from_env());
}

fn linsub_budget_from_env() -> Option<usize> {
    const DEFAULT: usize = 16;
    match std::env::var("Y_ZK_LINSUB_BUDGET") {
        Err(_) => Some(DEFAULT),
        Ok(v) if v.eq_ignore_ascii_case("off") => None,
        Ok(v) => v.trim().parse().ok().or(Some(DEFAULT)),
    }
}

// `Y_ZK_CSE=off` disables common-subexpression elimination.
//
// It exists because the pass's cost and its benefit are wildly different
// shapes, and neither was measurable before. On a 1000-hash Poseidon chain it
// is **38% of compile time and removes 1.25% of the constraints**; on the
// sparse dot-product circuit it removes *nothing at all* and still costs. That
// is not an argument for turning it off by default — ZK circuits are compiled
// once and proved many times, so a permanent 1.25% off every future proof beats
// a one-off 38% of one compile after enough proofs — but it is an argument for
// being able to measure the trade rather than assume it.
//
// Off is not a correctness risk: CSE only merges constraints that are already
// identical, so the unreduced circuit accepts exactly the same witnesses.
thread_local! {
    static CSE_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

fn cse_enabled() -> bool {
    CSE_ENABLED.with(|c| c.get())
}

/// Enable or disable CSE for this thread. Thread-local for the same reason
/// `LINSUB_BUDGET` is: a test must be able to compile the same circuit both ways
/// without racing the rest of the suite.
pub fn set_cse_enabled(on: bool) {
    CSE_ENABLED.with(|c| c.set(on));
}

pub fn init_cse_from_env() {
    if let Ok(v) = std::env::var("Y_ZK_CSE") {
        if v.eq_ignore_ascii_case("off") || v == "0" {
            set_cse_enabled(false);
        }
    }
}

/// Set the fill-in budget for this thread. `None` turns the pass off entirely,
/// which is only useful as a differential baseline - the reduced circuit and the
/// unreduced one must accept exactly the same witnesses.
pub fn set_linsub_budget(budget: Option<usize>) {
    LINSUB_BUDGET.with(|b| b.set(budget));
}

// `Y_ZK_COMPACT=off` disables wire compaction.
//
// `substitute_linear_constraints` and `dedup_identical_products` both leave
// wires behind: the first deletes the constraint that defined one, the second
// points two products at a single wire and abandons the loser. Neither renumbers
// the survivors, because doing so mid-fixpoint would invalidate every index the
// round is holding. So the wire count stayed at its pre-reduction value, and
// Groth16 pays per wire - which is the entire gap between Y emitting 1.86x fewer
// constraints than circom and only proving 1.4x faster.
//
// Off is a differential baseline, not a safety valve: compaction only drops
// wires that no surviving constraint mentions and no verifier sees, so the
// compacted circuit and the uncompacted one accept the same assignments modulo
// the renumbering.
thread_local! {
    static COMPACT_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

fn compaction_enabled() -> bool {
    COMPACT_ENABLED.with(|c| c.get())
}

/// Enable or disable wire compaction for this thread. Thread-local for the same
/// reason `LINSUB_BUDGET` is: a test must be able to compile the same circuit
/// both ways without racing the rest of the suite.
pub fn set_wire_compaction(on: bool) {
    COMPACT_ENABLED.with(|c| c.set(on));
}

pub fn init_compaction_from_env() {
    if let Ok(v) = std::env::var("Y_ZK_COMPACT") {
        if v.eq_ignore_ascii_case("off") || v == "0" {
            set_wire_compaction(false);
        }
    }
}

fn linsub_fill_budget() -> Option<usize> {
    LINSUB_BUDGET.with(|b| b.get())
}

/// Replace each eliminated wire by its defining expression.
///
/// `sub_idx` is dense and indexed by wire id, `u32::MAX` where nothing changed,
/// for the reason given on `replace_wires_in_lc`: this runs once per term over
/// the whole circuit, and a hash probe for an overwhelmingly likely miss is the
/// cost of the pass.
fn substitute_lc(lc: &mut LinearCombination, sub_idx: &[u32], exprs: &[LinearCombination]) {
    let hit = lc
        .terms
        .iter()
        .any(|(w, _)| matches!(sub_idx.get(*w), Some(&i) if i != u32::MAX));
    if !hit {
        return;
    }
    let mut out = LinearCombination::zero();
    for (w, coeff) in &lc.terms {
        match sub_idx.get(*w).copied() {
            Some(i) if i != u32::MAX => out.add_linear(&exprs[i as usize], *coeff),
            _ => out.add_term(*w, *coeff),
        }
    }
    out.simplify();
    *lc = out;
}

/// Replace a single wire by an expression, in place. Returns whether the wire
/// was there at all.
fn substitute_one(lc: &mut LinearCombination, wire: usize, expr: &LinearCombination) -> bool {
    let Some(pos) = lc.terms.iter().position(|(w, _)| *w == wire) else {
        return false;
    };
    let coeff = lc.terms[pos].1;
    lc.terms.swap_remove(pos);
    lc.add_linear(expr, coeff);
    lc.is_simplified = false;
    lc.simplify();
    true
}

/// Whether a recipe refers to wires only through `LinearCombination`s.
///
/// `substitute_linear_constraints` can rewrite an LC into a longer expression;
/// it cannot rewrite a bare `SignalId`, because there is nowhere to put one. The
/// flat variants are sorted into those two groups by hand at the match in that
/// function. `IfZeroLc` nests, so its group depends on what is inside it, and
/// this answers that question rather than assuming the front end only ever
/// builds LC-only branches - which is true today and is not a property anything
/// enforces.
fn witness_op_is_lc_only(op: &WitnessOp) -> bool {
    match op {
        WitnessOp::Const(_)
        | WitnessOp::Unknown
        | WitnessOp::LoadInput { .. }
        | WitnessOp::IsZeroLc(_)
        | WitnessOp::InvOrZeroLc(_)
        | WitnessOp::BitOfLc { .. }
        | WitnessOp::IntDivLc(..)
        | WitnessOp::IntModLc(..)
        | WitnessOp::MulLc(..)
        | WitnessOp::DivLc(..)
        | WitnessOp::MulAddLc(..) => true,
        WitnessOp::Add(..)
        | WitnessOp::Sub(..)
        | WitnessOp::Mul(..)
        | WitnessOp::Div(..)
        | WitnessOp::Inv(..)
        | WitnessOp::AssertEq(..)
        | WitnessOp::HintBlock { .. } => false,
        WitnessOp::IfZeroLc(_, then_, else_) => {
            witness_op_is_lc_only(then_) && witness_op_is_lc_only(else_)
        }
    }
}

/// Rewrite the linear combinations a recipe holds.
///
/// Exhaustive, no `_ =>` arm, exactly as in `remap_witness_op`. The `SignalId`
/// variants are no-ops here and that is safe only because
/// `substitute_linear_constraints` refuses to eliminate any wire one of them
/// mentions - if that guard is ever removed, these arms become silent
/// corruption rather than a compile error, so the two must be read together.
fn substitute_witness_op(op: &mut WitnessOp, sub_idx: &[u32], exprs: &[LinearCombination]) {
    match op {
        WitnessOp::Const(_)
        | WitnessOp::Unknown
        | WitnessOp::LoadInput { .. }
        | WitnessOp::Add(..)
        | WitnessOp::Sub(..)
        | WitnessOp::Mul(..)
        | WitnessOp::Div(..)
        | WitnessOp::Inv(..)
        | WitnessOp::AssertEq(..)
        | WitnessOp::HintBlock { .. } => {}
        WitnessOp::IsZeroLc(lc) | WitnessOp::InvOrZeroLc(lc) | WitnessOp::BitOfLc { lc, .. } => {
            substitute_lc(lc, sub_idx, exprs)
        }
        WitnessOp::IntDivLc(a, b)
        | WitnessOp::IntModLc(a, b)
        | WitnessOp::MulLc(a, b)
        | WitnessOp::DivLc(a, b) => {
            substitute_lc(a, sub_idx, exprs);
            substitute_lc(b, sub_idx, exprs);
        }
        WitnessOp::MulAddLc(a, b, c) => {
            substitute_lc(a, sub_idx, exprs);
            substitute_lc(b, sub_idx, exprs);
            substitute_lc(c, sub_idx, exprs);
        }
        WitnessOp::IfZeroLc(cond, then_, else_) => {
            substitute_lc(cond, sub_idx, exprs);
            substitute_witness_op(then_, sub_idx, exprs);
            substitute_witness_op(else_, sub_idx, exprs);
        }
    }
}

/// `A * B = C` where all three are constants and the identity holds.
///
/// Only ever used to DROP a constraint, so it must be conservative in one
/// direction only: a constraint it misclassifies as vacuous is a constraint
/// deleted from the statement being proved.
fn constraint_is_vacuous(c: &Constraint) -> bool {
    match (c.a.is_constant(), c.b.is_constant(), c.c.is_constant()) {
        (Some(a), Some(b), Some(cc)) => a.mul(&b) == cc,
        // `0 * <anything> = 0` holds whatever the other side is.
        (Some(a), _, Some(cc)) if a.is_zero() && cc.is_zero() => true,
        (_, Some(b), Some(cc)) if b.is_zero() && cc.is_zero() => true,
        _ => false,
    }
}

#[inline]
fn replace_signal(s: &mut SignalId, subst: &[usize]) {
    if let Some(&new_w) = subst.get(s.0) {
        s.0 = new_w;
    }
}

/// Rewrite every wire a `WitnessOp` refers to.
///
/// Matched exhaustively ON PURPOSE - no `_ =>` arm. A recipe that quietly keeps
/// a stale wire evaluates it as zero and makes a satisfiable circuit
/// unwitnessable, with nothing but `satisfied = false` to show for it. Adding a
/// variant must be a compile error here, not a silent omission. This is the same
/// rule CLAUDE.md states for unhandled AST nodes in soundness-critical passes.
fn remap_witness_op(op: &mut WitnessOp, subst: &[usize]) {
    match op {
        WitnessOp::Const(_) | WitnessOp::Unknown => {}
        // `input_idx` indexes the caller's input slice, not the wire table;
        // `execute_host_witness_ir` consumes those positionally.
        WitnessOp::LoadInput { .. } => {}
        WitnessOp::Add(a, b)
        | WitnessOp::Sub(a, b)
        | WitnessOp::Mul(a, b)
        | WitnessOp::Div(a, b)
        | WitnessOp::AssertEq(a, b) => {
            replace_signal(a, subst);
            replace_signal(b, subst);
        }
        WitnessOp::Inv(a) => replace_signal(a, subst),
        WitnessOp::HintBlock { inputs, outputs, ops } => {
            for s in inputs.iter_mut().chain(outputs.iter_mut()) {
                replace_signal(s, subst);
            }
            for hop in ops {
                match hop {
                    HintOp::NonDeterministicInv { src, dst } | HintOp::AssignExpr { dst, src } => {
                        replace_signal(src, subst);
                        replace_signal(dst, subst);
                    }
                    HintOp::BitDecompose { src, dst_bits } => {
                        replace_signal(src, subst);
                        for b in dst_bits {
                            replace_signal(b, subst);
                        }
                    }
                }
            }
        }
        WitnessOp::IsZeroLc(lc) | WitnessOp::InvOrZeroLc(lc) | WitnessOp::BitOfLc { lc, .. } => {
            replace_wires_in_lc(lc, subst)
        }
        WitnessOp::IntDivLc(a, b)
        | WitnessOp::IntModLc(a, b)
        | WitnessOp::MulLc(a, b)
        | WitnessOp::DivLc(a, b) => {
            replace_wires_in_lc(a, subst);
            replace_wires_in_lc(b, subst);
        }
        WitnessOp::MulAddLc(a, b, c) => {
            replace_wires_in_lc(a, subst);
            replace_wires_in_lc(b, subst);
            replace_wires_in_lc(c, subst);
        }
        // Recursive: a branch is itself a recipe, and a stale wire inside one
        // is exactly as invisible as a stale wire at the top level.
        WitnessOp::IfZeroLc(cond, then_, else_) => {
            replace_wires_in_lc(cond, subst);
            remap_witness_op(then_, subst);
            remap_witness_op(else_, subst);
        }
    }
}

/// circomlib's t=3 Poseidon parameters, parsed once.
///
/// The tables in `zk_poseidon_constants.rs` are hex STRINGS - 144 of them, 64
/// digits each - and `emit_poseidon` used to parse every one on every call.
/// `BigUint::from_hex_str` is a mul-and-add per digit and each step allocates,
/// so a single `poseidon_hash(a, b)` spent ~37,000 allocations reconstructing
/// values that are compile-time constants. On a 1000-hash chain that was ~40%
/// of everything the emitter allocated.
///
/// Keyed on the active modulus because `Fr` is stored in Montgomery form, which
/// is defined relative to it: reusing a cache entry across a field switch would
/// silently reinterpret every constant. `emit_poseidon` independently refuses
/// any field but BN254, so in practice the key never changes - it is here so
/// that fact does not have to be re-derived if that ever loosens.
pub struct PoseidonT3Params {
    pub c: Vec<Fr>,
    pub s_sparse: Vec<Fr>,
    pub m: [[Fr; 3]; 3],
    pub p: [[Fr; 3]; 3],
}

thread_local! {
    static POSEIDON_T3_CACHE: RefCell<Option<(BigUint, Rc<PoseidonT3Params>)>> =
        const { RefCell::new(None) };
}

fn poseidon_t3_params() -> Result<Rc<PoseidonT3Params>, String> {
    let modulus = active_modulus();
    if let Some(hit) = POSEIDON_T3_CACHE.with(|cell| {
        cell.borrow()
            .as_ref()
            .filter(|(m, _)| *m == modulus)
            .map(|(_, p)| Rc::clone(p))
    }) {
        return Ok(hit);
    }

    let hex = |s: &str| -> Result<Fr, String> { BigUint::from_hex_str(s).map(Fr::from_biguint) };
    let c: Vec<Fr> = POSEIDON_C_T3.iter().map(|s| hex(s)).collect::<Result<_, _>>()?;
    let s_sparse: Vec<Fr> = POSEIDON_S_T3.iter().map(|s| hex(s)).collect::<Result<_, _>>()?;
    let mut m = [[Fr::zero(); 3]; 3];
    let mut p = [[Fr::zero(); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = hex(POSEIDON_M_T3[i][j])?;
            p[i][j] = hex(POSEIDON_P_T3[i][j])?;
        }
    }

    let params = Rc::new(PoseidonT3Params { c, s_sparse, m, p });
    POSEIDON_T3_CACHE.with(|cell| *cell.borrow_mut() = Some((modulus, Rc::clone(&params))));
    Ok(params)
}

pub struct ZkEmitter {
    pub variables: Vec<String>,
    pub public_inputs: Vec<usize>,
    pub private_inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub constraints: Vec<Constraint>,
    pub next_var_id: usize,

    // Scope management: Maps variables to their bound representation
    scopes: Vec<HashMap<String, WireBinding>>,
    // Constant bindings for static loop evaluation
    const_bindings: HashMap<String, Fr>,
    // Tracker for active calls to reject recursive loops
    active_calls: Vec<String>,
    pub active_field: ScalarField,
    pub active_scheme: ProofScheme,
    // Track unconstrained hint variables to enforce sound R1CS matrix constraints (error[Z0042])
    pub unconstrained_hint_vars: HashMap<usize, (String, Span)>,
    /// Explicit witness recipes for wires that `build_witness_ir`'s constraint
    /// scan cannot reconstruct - see the linear-combination `WitnessOp`
    /// variants. Without these, gadget wires stay unsolved and the circuit
    /// cannot be proved.
    witness_recipes: HashMap<usize, WitnessOp>,
}

fn expr_references_var(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Ident(n, _) => n == name,
        Expr::BinaryOp { left, right, .. } => {
            expr_references_var(left, name) || expr_references_var(right, name)
        }
        Expr::Call { args, .. } => {
            args.iter().any(|arg| expr_references_var(arg, name))
        }
        _ => false,
    }
}

impl ZkEmitter {
    pub fn new() -> Self {
        init_cse_from_env();
        init_compaction_from_env();
        Self {
            variables: vec!["const_1".to_string()], // wire 0 is constant 1
            public_inputs: Vec::new(),
            private_inputs: Vec::new(),
            outputs: Vec::new(),
            constraints: Vec::new(),
            next_var_id: 1,
            scopes: vec![HashMap::new()],
            const_bindings: HashMap::new(),
            active_calls: Vec::new(),
            active_field: ScalarField::Bn254,
            active_scheme: ProofScheme::R1cs,
            unconstrained_hint_vars: HashMap::new(),
            witness_recipes: HashMap::new(),
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) {
        if !self.unconstrained_hint_vars.is_empty() {
            for (wire, _) in &constraint.a.terms {
                self.unconstrained_hint_vars.remove(wire);
            }
            for (wire, _) in &constraint.b.terms {
                self.unconstrained_hint_vars.remove(wire);
            }
            for (wire, _) in &constraint.c.terms {
                self.unconstrained_hint_vars.remove(wire);
            }
        }
        self.constraints.push(constraint);
    }

    fn new_wire(&mut self, name: &str) -> usize {
        let id = self.next_var_id;
        self.next_var_id += 1;
        self.variables.push(format!("{}_{}", name, id));
        id
    }

    /// Bit-decomposes `value` into `n_bits` boolean wires, LSB first, and
    /// constrains their recomposition back to `value`.
    ///
    /// This is the classic `Num2Bits`, and it does two jobs at once: it hands
    /// back the bits, and - because every bit is constrained boolean and they
    /// must recompose exactly - it PROVES `0 <= value < 2^n_bits`. That range
    /// proof is the part a sound field comparison cannot do without. Cost is
    /// `n_bits + 1` constraints.
    fn emit_num2bits(&mut self, value: &LinearCombination, n_bits: u32, span: &Span) -> Vec<usize> {
        let mut bits = Vec::with_capacity(n_bits as usize);
        let mut recomposed = LinearCombination::zero();
        let mut coeff = Fr::one();
        let two = Fr::from_u64(2);

        for i in 0..n_bits {
            let b = self.new_wire(&format!("bit{}", i));
            // The witness pass cannot recover a bit from the constraints (b*b=b
            // has two satisfying values), so record how to compute it.
            self.witness_recipes
                .insert(b, WitnessOp::BitOfLc { lc: value.clone(), bit: i });
            // Booleanity: b * b = b, satisfied only by 0 and 1.
            self.add_constraint(Constraint {
                a: LinearCombination::variable(b),
                b: LinearCombination::variable(b),
                c: LinearCombination::variable(b),
                span: Some(span.clone()),
            });
            recomposed.add_term(b, coeff.clone());
            coeff = coeff.mul(&two);
            bits.push(b);
        }

        // Recomposition: (sum 2^i * b_i) * 1 = value.
        self.add_constraint(Constraint {
            a: recomposed,
            b: LinearCombination::constant(Fr::one()),
            c: value.clone(),
            span: Some(span.clone()),
        });

        bits
    }

    /// `a < b` as a boolean linear combination, for operands that fit in
    /// `n_bits` bits.
    ///
    /// A field has no order, so "less than" is only meaningful once both
    /// operands are pinned to a bounded integer range - otherwise a prover can
    /// pick `a = p-1` and any ordering claim is vacuous. So both operands are
    /// range-checked first, then the standard trick: `a + 2^n - b` lies in
    /// `[1, 2^(n+1))` and therefore never wraps, and its top bit is 0 exactly
    /// when `a < b`.
    ///
    /// Cost is roughly `3n + 6` constraints (two operand range checks plus one
    /// `n+1`-bit decomposition) - about 102 at `n = 32`. That is inherent to
    /// comparison in R1CS, not an inefficiency here: circom's `LessThan` pays
    /// the same shape of cost, and the previous 3-constraint version of this
    /// operator was cheap only because it was computing something else
    /// entirely.
    ///
    /// If an operand does not fit in `n_bits`, its range check is unsatisfiable
    /// and no proof can be produced. That is fail-closed - a prover cannot slip
    /// an out-of-range value past it.
    fn emit_less_than(
        &mut self,
        a: &LinearCombination,
        b: &LinearCombination,
        n_bits: u32,
        span: &Span,
    ) -> LinearCombination {
        self.emit_num2bits(a, n_bits, span);
        self.emit_num2bits(b, n_bits, span);

        // diff = a + 2^n - b
        let mut diff = a.clone();
        diff.add_constant(Fr::from_u64(1u64 << n_bits));
        diff.add_linear(b, Fr::from_u64(0).sub(&Fr::one()));

        let bits = self.emit_num2bits(&diff, n_bits + 1, span);
        let top = bits[n_bits as usize];

        // a < b  <=>  top bit clear  =>  out = 1 - top
        let mut out = LinearCombination::constant(Fr::one());
        out.add_term(top, Fr::from_u64(0).sub(&Fr::one()));
        out
    }

    /// Force a boolean linear combination to be exactly 1.
    fn constrain_true(&mut self, lc: &LinearCombination, span: &Span) {
        self.add_constraint(Constraint {
            a: lc.clone(),
            b: LinearCombination::constant(Fr::one()),
            c: LinearCombination::constant(Fr::one()),
            span: Some(span.clone()),
        });
    }

    /// Constrain a linear combination to be 0 or 1, and return it.
    fn constrain_boolean(&mut self, lc: &LinearCombination, span: &Span) {
        self.add_constraint(Constraint {
            a: lc.clone(),
            b: lc.clone(),
            c: lc.clone(),
            span: Some(span.clone()),
        });
    }

    /// Integer `(quotient, remainder)` of `a / b`, both range-checked.
    ///
    /// The obvious encoding, `q * b = a - r` with `r < b`, is UNSOUND on its
    /// own, and the hole is worth stating because it is invisible: field
    /// division always succeeds. For `a = 7, b = 2` a prover can claim `r = 0`
    /// and supply `q = 7 * 2^-1 mod p`, an enormous field element. The
    /// constraint holds, `r < b` holds, and the circuit has proved
    /// `7 mod 2 == 0`.
    ///
    /// Range-checking `q` closes it. With `q < 2^n` and `b < 2^n` the product
    /// `q*b` is below `2^2n`, far under the modulus at `n = 32`, so the field
    /// equation forces the integer one and `q` must be the true quotient.
    ///
    /// Division by zero is rejected rather than defined: `b = 0` collapses the
    /// constraint to `r = a`, and `r < 0` is unsatisfiable. No proof exists,
    /// which is the correct outcome for `x / 0`.
    ///
    /// Cost is about `4n + 8` constraints (~136 at n=32). Integer division is
    /// genuinely expensive in R1CS - it is three range checks and a comparison.
    fn emit_int_div_mod(
        &mut self,
        a: &LinearCombination,
        b: &LinearCombination,
        n_bits: u32,
        span: &Span,
    ) -> (LinearCombination, LinearCombination) {
        let q = self.new_wire("intdiv_q");
        let r = self.new_wire("intdiv_r");
        self.witness_recipes
            .insert(q, WitnessOp::IntDivLc(a.clone(), b.clone()));
        self.witness_recipes
            .insert(r, WitnessOp::IntModLc(a.clone(), b.clone()));
        let q_lc = LinearCombination::variable(q);
        let r_lc = LinearCombination::variable(r);

        // q < 2^n. Without this the quotient can be any field element.
        self.emit_num2bits(&q_lc, n_bits, span);

        // q * b = a - r
        let mut rhs = a.clone();
        rhs.add_linear(&r_lc, Fr::from_u64(0).sub(&Fr::one()));
        rhs.simplify();
        self.add_constraint(Constraint {
            a: q_lc.clone(),
            b: b.clone(),
            c: rhs,
            span: Some(span.clone()),
        });

        // r < b, which also range-checks both r and b.
        let lt = self.emit_less_than(&r_lc, b, n_bits, span);
        self.constrain_true(&lt, span);

        (q_lc, r_lc)
    }

    /// Bitwise op over `n_bits`, returned as a linear combination.
    ///
    /// Both operands are decomposed (which range-checks them), combined bit by
    /// bit, and recomposed. The recomposition is free: it is a weighted sum, so
    /// only the per-bit products cost constraints. About `3n + 2` total.
    fn emit_bitwise(
        &mut self,
        a: &LinearCombination,
        b: &LinearCombination,
        op: &BinaryOp,
        n_bits: u32,
        span: &Span,
    ) -> LinearCombination {
        let a_bits = self.emit_num2bits(a, n_bits, span);
        let b_bits = self.emit_num2bits(b, n_bits, span);

        let neg_one = Fr::from_u64(0).sub(&Fr::one());
        let neg_two = Fr::from_u64(0).sub(&Fr::from_u64(2));
        let two = Fr::from_u64(2);

        let mut out = LinearCombination::zero();
        let mut coeff = Fr::one();
        for i in 0..n_bits as usize {
            let ai = LinearCombination::variable(a_bits[i]);
            let bi = LinearCombination::variable(b_bits[i]);
            // Every variant needs the AND of the two bits.
            let and = self.emit_mul_lc(&ai, &bi, "bitop_and");

            // AND -> ab;  OR -> a + b - ab;  XOR -> a + b - 2ab
            let mut bit = match op {
                BinaryOp::BitAnd => and,
                BinaryOp::BitOr | BinaryOp::BitXor => {
                    let mut t = ai;
                    t.add_linear(&bi, Fr::one());
                    let scale = if matches!(op, BinaryOp::BitOr) { neg_one.clone() } else { neg_two.clone() };
                    t.add_linear(&and, scale);
                    t
                }
                _ => unreachable!("emit_bitwise called with {:?}", op),
            };
            bit.simplify();
            out.add_linear(&bit, coeff.clone());
            coeff = coeff.mul(&two);
        }
        out.simplify();
        out
    }

    /// Shift by a compile-time constant, truncated to `n_bits`.
    ///
    /// Only constant shift amounts are supported. A variable shift is a
    /// multiplexer over every possible amount, which is a different gadget with
    /// a very different cost; emitting something cheaper that only works for
    /// the constant case would be a silent trap.
    fn emit_shift(
        &mut self,
        a: &LinearCombination,
        amount: u64,
        left: bool,
        n_bits: u32,
        span: &Span,
    ) -> LinearCombination {
        let bits = self.emit_num2bits(a, n_bits, span);
        let n = n_bits as u64;
        let mut out = LinearCombination::zero();
        if amount >= n {
            // Everything shifted out. Still emit the decomposition above so the
            // operand stays range-checked.
            return out;
        }
        let two = Fr::from_u64(2);
        for i in 0..n {
            // Source bit for output position `i`.
            let src = if left {
                if i < amount { continue; } else { i - amount }
            } else {
                let s = i + amount;
                if s >= n { continue; }
                s
            };
            let mut coeff = Fr::one();
            for _ in 0..i {
                coeff = coeff.mul(&two);
            }
            out.add_term(bits[src as usize], coeff);
        }
        out.simplify();
        out
    }

    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    fn bind(&mut self, name: &str, binding: WireBinding) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), binding);
        }
    }

    fn lookup(&self, name: &str) -> Option<WireBinding> {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.get(name) {
                return Some(binding.clone());
            }
        }
        None
    }

    fn bind_update(&mut self, name: &str, binding: WireBinding) -> bool {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), binding);
                return true;
            }
        }
        false
    }

    fn take_binding_from_scope(&mut self, name: &str) -> Option<(usize, WireBinding)> {
        for (idx, scope) in self.scopes.iter_mut().enumerate().rev() {
            if let Some(binding) = scope.remove(name) {
                return Some((idx, binding));
            }
        }
        None
    }

    fn bind_to_scope(&mut self, scope_idx: usize, name: &str, binding: WireBinding) {
        if scope_idx < self.scopes.len() {
            self.scopes[scope_idx].insert(name.to_string(), binding);
        }
    }

    fn lookup_in_scopes(&self, scopes_stack: &[HashMap<String, WireBinding>], start_idx: usize, name: &str) -> Option<LinearCombination> {
        for scope in scopes_stack[..=start_idx].iter().rev() {
            if let Some(binding) = scope.get(name) {
                return Some(match binding {
                    WireBinding::Wire(w) => LinearCombination::variable(*w),
                    WireBinding::Linear(lc) => lc.clone(),
                });
            }
        }
        None
    }

    fn lookup_const(&self, name: &str) -> Option<Fr> {
        self.const_bindings.get(name).cloned()
    }

    pub fn build_witness_ir(&self) -> WitnessIRGraph {
        let num_signals = self.variables.len();
        let mut nodes = Vec::with_capacity(num_signals);
        let mut signal_names = HashMap::new();

        nodes.push(WitnessOp::Const(Fr::one()));
        signal_names.insert(0, "const_1".to_string());

        // Index the `a*b = c` constraints by their single output wire, ONCE.
        //
        // This loop used to be nested inside the per-variable loop below - a
        // full scan of every constraint for every variable, i.e. O(V*C). It is
        // quadratic in the circuit size and it showed: building the witness IR
        // for the polynomial circuit cost 0.06s at 10k constraints but 5.96s at
        // 100k, each doubling costing ~4x. Emitting the R1CS itself is linear,
        // so this one function was the entire super-linear term in the pipeline.
        // `or_insert` keeps the FIRST matching constraint, preserving the
        // original "break on first match" semantics exactly.
        //
        // All three coefficients must be 1. `WitnessOp::Mul` multiplies two
        // WIRES and has nowhere to carry a scale factor, so `2a * b = t` would
        // be reconstructed as `a * b` - the wrong value, with no error. It was
        // survivable only because such a wire then fails the forward pass's
        // satisfiability check and gets rediscovered by the back-propagation
        // sweep, which is the 0.4-seconds-for-one-wire path described below.
        // Wires that fail this test fall through to `lc_by_output`, which keeps
        // the coefficients.
        let one = Fr::one();
        let mut mul_by_output: HashMap<usize, (usize, usize)> = HashMap::new();
        for c in &self.constraints {
            if c.c.terms.len() == 1
                && c.a.terms.len() == 1
                && c.b.terms.len() == 1
                && c.a.terms[0].1 == one
                && c.b.terms[0].1 == one
                && c.c.terms[0].1 == one
            {
                mul_by_output
                    .entry(c.c.terms[0].0)
                    .or_insert((c.a.terms[0].0, c.b.terms[0].0));
            }
        }

        // Second chance for wires the single-term scan above cannot see.
        //
        // R1CS allows `A` and `B` to be arbitrary linear combinations, and the
        // emitter uses that freely - binding `main`'s return value emits
        // `<dense lc> * 1 = out`, and Poseidon squares a whole linear
        // combination without allocating a wire for it. None of those match the
        // one-term-per-side pattern, so their outputs came back `Unknown` and
        // were left to back-propagation. That is correct but ruinous: a single
        // unknown wire makes the forward pass fail its satisfiability check,
        // which forfeits the early exit and runs the full sweep. One Poseidon
        // hash spent 0.002s in the forward pass and then 0.427s rediscovering
        // exactly one wire.
        //
        // Only wires the first scan missed are entered here, so the common case
        // (a million tiny `a*b=c` constraints) does not pay to clone any linear
        // combinations.
        //
        // The guard is that the constraint must not mention its own output: a
        // recipe that reads the wire it defines reads a zero and looks solved.
        // It used to be the stronger `every wire < out`, which was how recipes
        // stayed evaluable back when the solver walked in wire-index order. It
        // does not any more - `topological_order` is a real Kahn sort - and the
        // stronger form excludes circom entirely, where `signal output out;` is
        // declared at the top of a template and assigned at the bottom, so its
        // constraint always references higher-numbered wires.
        let mut lc_by_output: HashMap<usize, (LinearCombination, LinearCombination)> = HashMap::new();
        for c in &self.constraints {
            if c.c.terms.len() != 1 || c.c.terms[0].1 != one {
                continue;
            }
            let out = c.c.terms[0].0;
            if out == 0 || mul_by_output.contains_key(&out) || lc_by_output.contains_key(&out) {
                continue;
            }
            if c.a.terms.iter().chain(c.b.terms.iter()).all(|(w, _)| *w != out) {
                lc_by_output.insert(out, (c.a.clone(), c.b.clone()));
            }
        }

        for (idx, name) in self.variables.iter().enumerate().skip(1) {
            signal_names.insert(idx, name.clone());
            if self.public_inputs.contains(&idx) {
                nodes.push(WitnessOp::LoadInput { input_idx: idx, is_public: true });
            } else if self.private_inputs.contains(&idx) {
                nodes.push(WitnessOp::LoadInput { input_idx: idx, is_public: false });
            } else if let Some(op) = self.witness_recipes.get(&idx) {
                nodes.push(op.clone());
            } else if let Some((a, b)) = mul_by_output.get(&idx) {
                nodes.push(WitnessOp::Mul(SignalId(*a), SignalId(*b)));
            } else if let Some((a, b)) = lc_by_output.get(&idx) {
                nodes.push(WitnessOp::MulLc(a.clone(), b.clone()));
            } else {
                nodes.push(WitnessOp::Unknown);
            }
        }

        let topological_order = Self::topological_order(&nodes);

        WitnessIRGraph {
            field: FieldType::Bn254,
            num_public_inputs: self.public_inputs.len(),
            num_private_inputs: self.private_inputs.len(),
            num_signals,
            nodes,
            signal_names,
            topological_order,
        }
    }

    /// Every wire a recipe reads.
    ///
    /// Exhaustive on purpose. A variant missing from here is not a compile
    /// error but a *silent* one: its dependencies look empty, the sort places
    /// it before the wires it needs, and it evaluates them as zero.
    fn witness_op_deps(op: &WitnessOp, out: &mut Vec<usize>) {
        let mut lc = |l: &LinearCombination| out.extend(l.terms.iter().map(|(w, _)| *w));
        match op {
            WitnessOp::Const(_) | WitnessOp::Unknown | WitnessOp::LoadInput { .. } => {}
            WitnessOp::Add(a, b)
            | WitnessOp::Sub(a, b)
            | WitnessOp::Mul(a, b)
            | WitnessOp::Div(a, b)
            | WitnessOp::AssertEq(a, b) => {
                out.push(a.0);
                out.push(b.0);
            }
            WitnessOp::Inv(a) => out.push(a.0),
            WitnessOp::HintBlock { inputs, .. } => out.extend(inputs.iter().map(|s| s.0)),
            WitnessOp::IsZeroLc(l) | WitnessOp::InvOrZeroLc(l) | WitnessOp::BitOfLc { lc: l, .. } => lc(l),
            WitnessOp::IntDivLc(a, b)
            | WitnessOp::IntModLc(a, b)
            | WitnessOp::MulLc(a, b)
            | WitnessOp::DivLc(a, b) => {
                lc(a);
                lc(b);
            }
            WitnessOp::MulAddLc(a, b, c) => {
                lc(a);
                lc(b);
                lc(c);
            }
            // BOTH branches, not just the one that will be taken. Which branch
            // runs is a runtime fact and this is a compile-time sort; if the
            // untaken branch's wires were left out of the ordering, a change of
            // input would evaluate them as zero.
            WitnessOp::IfZeroLc(cond, then_, else_) => {
                out.extend(cond.terms.iter().map(|(w, _)| *w));
                Self::witness_op_deps(then_, out);
                Self::witness_op_deps(else_, out);
            }
        }
    }

    /// Every wire a recipe *writes* besides the one it is filed under.
    ///
    /// Only `HintBlock` has any: it computes several wires in one go, so if one
    /// of them survives compaction the rest must too - a block whose outputs are
    /// half-deleted would write past the end of the witness. Exhaustive for the
    /// same reason `witness_op_deps` is, and paired with it at every call site.
    fn witness_op_writes(op: &WitnessOp, out: &mut Vec<usize>) {
        match op {
            WitnessOp::Const(_)
            | WitnessOp::Unknown
            | WitnessOp::LoadInput { .. }
            | WitnessOp::Add(..)
            | WitnessOp::Sub(..)
            | WitnessOp::Mul(..)
            | WitnessOp::Div(..)
            | WitnessOp::AssertEq(..)
            | WitnessOp::Inv(_)
            | WitnessOp::IsZeroLc(_)
            | WitnessOp::InvOrZeroLc(_)
            | WitnessOp::BitOfLc { .. }
            | WitnessOp::IntDivLc(..)
            | WitnessOp::IntModLc(..)
            | WitnessOp::MulLc(..)
            | WitnessOp::DivLc(..)
            | WitnessOp::MulAddLc(..) => {}
            WitnessOp::IfZeroLc(_, then_, else_) => {
                Self::witness_op_writes(then_, out);
                Self::witness_op_writes(else_, out);
            }
            WitnessOp::HintBlock { outputs, ops, .. } => {
                out.extend(outputs.iter().map(|s| s.0));
                for hop in ops {
                    match hop {
                        HintOp::NonDeterministicInv { dst, .. } | HintOp::AssignExpr { dst, .. } => {
                            out.push(dst.0)
                        }
                        HintOp::BitDecompose { dst_bits, .. } => {
                            out.extend(dst_bits.iter().map(|s| s.0))
                        }
                    }
                }
            }
        }
    }

    /// A real dependency order for the forward witness pass.
    ///
    /// This field was `(0..num_signals)` — the identity — while being called
    /// `topological_order`, and the solver ignored it and walked by wire index
    /// instead. That works only when every recipe happens to reference
    /// lower-numbered wires, which is true of Y's own emitter because it
    /// allocates a wire at the moment it defines it. It is NOT true of circom,
    /// where `signal output out;` is conventionally declared at the top of a
    /// template and assigned at the bottom: `out`'s recipe then reads wires
    /// numbered above it, the forward pass evaluates them as zero, and the
    /// witness silently fails to satisfy the circuit it came from.
    ///
    /// Kahn's algorithm; anything left in a cycle keeps index order and is left
    /// for the back-propagation sweep, which is what handles it today anyway.
    fn topological_order(nodes: &[WitnessOp]) -> Vec<SignalId> {
        let n = nodes.len();
        let mut deps: Vec<Vec<usize>> = Vec::with_capacity(n);
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut indegree = vec![0usize; n];
        let mut scratch = Vec::new();

        for (i, op) in nodes.iter().enumerate() {
            scratch.clear();
            Self::witness_op_deps(op, &mut scratch);
            scratch.retain(|w| *w != i && *w < n);
            scratch.sort_unstable();
            scratch.dedup();
            indegree[i] = scratch.len();
            for &d in &scratch {
                dependents[d].push(i);
            }
            deps.push(scratch.clone());
        }

        let mut ready: Vec<usize> = (0..n).filter(|i| indegree[*i] == 0).collect();
        ready.reverse(); // pop() yields ascending index, keeping output stable
        let mut order = Vec::with_capacity(n);
        let mut emitted = vec![false; n];
        while let Some(i) = ready.pop() {
            order.push(SignalId(i));
            emitted[i] = true;
            for &d in &dependents[i] {
                indegree[d] -= 1;
                if indegree[d] == 0 {
                    ready.push(d);
                }
            }
        }
        // Cyclic remainder, in index order.
        for i in 0..n {
            if !emitted[i] {
                order.push(SignalId(i));
            }
        }
        order
    }

    // ────────────────────────────────────────────────────────
    // Compilation Entry Points
    // ────────────────────────────────────────────────────────

    pub fn emit_program(&mut self, prog: &Program) -> Result<String, String> {
        // Flatten items recursively (entering modules) to allow seamless function lookups
        let mut flat_items = Vec::new();
        fn flatten_items(items: &[Item], dest: &mut Vec<Item>) {
            for item in items {
                dest.push(item.clone());
                if let Item::Module(m) = item {
                    flatten_items(&m.items, dest);
                }
            }
        }
        flatten_items(&prog.items, &mut flat_items);

        // Scan for @zk_target module attribute
        let mut active_field = ScalarField::Bn254;
        let mut active_scheme = ProofScheme::R1cs;
        for item in &prog.items {
            if let Item::Module(m) = item {
                if let Some(zk_target) = &m.zk_target {
                    active_field = match zk_target.field {
                        ScalarFieldEnum::Bn254 => ScalarField::Bn254,
                        ScalarFieldEnum::Bls12_381 => ScalarField::Bls12_381,
                        ScalarFieldEnum::Pallas => ScalarField::Pallas,
                        ScalarFieldEnum::Vesta => ScalarField::Vesta,
                    };
                    active_scheme = match zk_target.scheme {
                        ProofSchemeEnum::R1cs => ProofScheme::R1cs,
                        ProofSchemeEnum::Plonkish => ProofScheme::Plonkish,
                    };
                    break;
                }
            }
        }

        // `scheme = "plonkish"` used to be accepted, stored, printed in the
        // `.r1cs.txt` header as `Proof Scheme: Plonkish`, and then IGNORED - the
        // emitter has exactly one arithmetization and produced R1CS regardless.
        // A user selecting a PLONKish backend got a clean compile, a success
        // message, a header claiming they had one, and an R1CS file.
        //
        // Refusing is the fix, not a stopgap. This repo has been here before
        // with `@ZeroDrift` (lexed, counted, printed, read by no backend) and
        // with the Hopper intrinsics that assembled to nothing: a named gap
        // costs a user five minutes, a silently-wrong artifact costs them
        // however long it takes to suspect the compiler. For a proof system it
        // is worse than that - the file would go to a PLONKish prover that
        // cannot read it, or worse, be assumed to carry PLONKish's soundness
        // properties.
        if matches!(active_scheme, ProofScheme::Plonkish) {
            return Err(
                "Circuit target error: `scheme = \"plonkish\"` is not implemented. Y emits \
                 Rank-1 Constraint Systems only.\n  \
                 hint: remove the `scheme` argument, or set `scheme = \"r1cs\"`, and prove with \
                 a Groth16 backend (snarkjs and arkworks both read Y's .r1cs/.wtns).\n  \
                 note: this previously compiled and emitted R1CS while reporting `Proof Scheme: \
                 Plonkish`. If you have artifacts produced that way, they are R1CS."
                    .to_string(),
            );
        }

        // Apply selected target configuration
        self.active_field = active_field;
        self.active_scheme = active_scheme;
        let config = FieldConfig::get(self.active_field);
        set_active_modulus(&config.p);

        // Collect all functions inside the flattened items list
        let mut target_func: Option<&FuncDecl> = None;
        for item in &flat_items {
            match item {
                Item::Func(f) => {
                    if f.name == "main" || f.name == "circuit" {
                        target_func = Some(f);
                    }
                }
                _ => {}
            }
        }

        let f = match target_func {
            Some(func) => func,
            None => return Err("No entry function 'main' or 'circuit' found for ZK Circuit target.".to_string()),
        };

        // `emit_program` used to be one opaque 92% of ZK compile time, which is
        // not a number anyone can act on. These two sub-phases are the split
        // that matters: building the constraints, versus the CSE pass over them.
        // Since the field arithmetic was fixed the second one is the larger of
        // the two (1.22 s against 0.60 s on a 1000-hash Poseidon chain), and
        // nothing would have said so without this line.
        let t_start = std::time::Instant::now();

        // Note: we must pass &flat_items here so all nested functions/structs are in scope
        self.emit_circuit_entry(f, &flat_items)?;
        let t_emitted = std::time::Instant::now();

        // Run optimization pass: dead-wire elimination & constraint reduction
        self.optimize_circuit();

        // `Y_ZK_COMPOSITION=1` breaks the emitter's live memory down by owner.
        //
        // Memory, not time, is the ceiling on circuit size here, and a peak-RSS
        // number alone says nothing about what to fix. This is what found the
        // `emit_mul_lc` duplication: `witness_recipes` was holding 1096 bytes
        // per constraint against the constraints' own 1135 - a second verbatim
        // copy of every Poseidon S-box's operands - which no amount of staring
        // at a total would have revealed. Cheap enough to leave in: one pass
        // over the constraints, only when the variable is set.
        if std::env::var("Y_ZK_COMPOSITION").is_ok() {
            let (mut len, mut cap, mut n_lc) = (0usize, 0usize, 0usize);
            for c in &self.constraints {
                for lc in [&c.a, &c.b, &c.c] {
                    len += lc.terms.len();
                    cap += lc.terms.capacity();
                    n_lc += 1;
                }
            }
            let (mut recipe_cap, mut recipe_lcs) = (0usize, 0usize);
            for op in self.witness_recipes.values() {
                let mut acc = |lc: &LinearCombination| {
                    recipe_cap += lc.terms.capacity();
                    recipe_lcs += 1;
                };
                match op {
                    WitnessOp::IsZeroLc(l) | WitnessOp::InvOrZeroLc(l) | WitnessOp::BitOfLc { lc: l, .. } => acc(l),
                    WitnessOp::IntDivLc(a, b)
                    | WitnessOp::IntModLc(a, b)
                    | WitnessOp::MulLc(a, b)
                    | WitnessOp::DivLc(a, b) => {
                        acc(a);
                        acc(b);
                    }
                    WitnessOp::MulAddLc(a, b, c) => {
                        acc(a);
                        acc(b);
                        acc(c);
                    }
                    _ => {}
                }
            }
            let nc = self.constraints.len().max(1) as f64;
            let term = std::mem::size_of::<(usize, Fr)>() as f64;
            eprintln!("[Y ZK COMPOSITION] {} constraints, {} linear combinations", self.constraints.len(), n_lc);
            eprintln!(
                "  constraint terms    {:>10} used / {:>10} allocated ({:.1}% slack)   {:>7.0} B/constraint",
                len, cap, 100.0 * (cap - len) as f64 / len.max(1) as f64, cap as f64 * term / nc
            );
            eprintln!(
                "  Constraint structs                                                    {:>7.0} B/constraint",
                std::mem::size_of::<Constraint>() as f64
            );
            eprintln!(
                "  witness_recipes     {:>10} recipes, {:>8} LCs                   {:>7.0} B/constraint",
                self.witness_recipes.len(), recipe_lcs, recipe_cap as f64 * term / nc
            );
            eprintln!(
                "  variable names      {:>10} wires                                  {:>7.0} B/constraint",
                self.variables.len(),
                self.variables.iter().map(|v| v.capacity() + std::mem::size_of::<String>()).sum::<usize>() as f64 / nc
            );
        }
        if std::env::var("Y_ZK_TIMING").is_ok() {
            eprintln!(
                "[Y ZK TIMING]   emit_circuit_entry   {:>8.3} s",
                (t_emitted - t_start).as_secs_f64()
            );
            eprintln!(
                "[Y ZK TIMING]   optimize_circuit     {:>8.3} s",
                t_emitted.elapsed().as_secs_f64()
            );
        }

        // Format R1CS Output
        let mut out = String::new();
        writeln!(&mut out, "=========================================================").unwrap();
        writeln!(&mut out, "   Y-lang Native ZK Circuit Target: Rank-1 Constraint System").unwrap();
        writeln!(&mut out, "=========================================================\n").unwrap();
        writeln!(&mut out, "Curve Field: {} (prime size: {} bits)", config.name.to_uppercase(), config.p.bit_len()).unwrap();
        writeln!(&mut out, "Modulus r: {}\n", config.p.to_decimal_string()).unwrap();
        writeln!(&mut out, "Proof Scheme: {:?}", self.active_scheme).unwrap();

        writeln!(&mut out, "Parameters:").unwrap();
        writeln!(&mut out, "  - Total wires (including intermediate): {}", self.next_var_id).unwrap();
        writeln!(&mut out, "  - Constraints: {}", self.constraints.len()).unwrap();
        writeln!(&mut out, "  - Public inputs: {:?}", self.public_inputs).unwrap();
        writeln!(&mut out, "  - Private inputs: {:?}", self.private_inputs).unwrap();
        writeln!(&mut out, "  - Outputs: {:?}\n", self.outputs).unwrap();

        if self.variables.len() <= 1000 {
            writeln!(&mut out, "Wire Assignments:").unwrap();
            for (i, name) in self.variables.iter().enumerate() {
                let role = if self.public_inputs.contains(&i) {
                    " [Public Input]"
                } else if self.private_inputs.contains(&i) {
                    " [Private Witness Input]"
                } else if self.outputs.contains(&i) {
                    " [Output]"
                } else if i == 0 {
                    " [Constant 1]"
                } else {
                    ""
                };
                writeln!(&mut out, "  w_{} = {}{}", i, name, role).unwrap();
            }
        } else {
            writeln!(&mut out, "Wire Assignments:\n  [Detailed wire assignments list omitted for circuits with > 1000 variables to optimize compilation performance]").unwrap();
        }

        writeln!(&mut out, "\nR1CS Equations (A * B = C):").unwrap();
        if self.constraints.len() <= 1000 {
            let var_map: HashMap<usize, String> = self.variables.iter().enumerate()
                .map(|(i, _name)| (i, format!("w_{}", i)))
                .collect();

            for (idx, c) in self.constraints.iter().enumerate() {
                writeln!(
                    &mut out,
                    "  Constraint #{}:\n    A: ({})\n    B: ({})\n    C: ({})",
                    idx + 1,
                    c.a.to_string(&var_map),
                    c.b.to_string(&var_map),
                    c.c.to_string(&var_map)
                ).unwrap();
                if let Some(ref span) = c.span {
                    writeln!(&mut out, "    Source Location: line {}, col {}", span.line, span.col).unwrap();
                }
                writeln!(&mut out, "").unwrap();
            }
        } else {
            writeln!(&mut out, "  [Detailed constraints list omitted for circuits with > 1000 constraints to optimize compilation performance]").unwrap();
        }

        Ok(out)
    }

    fn emit_circuit_entry(&mut self, f: &FuncDecl, items: &[Item]) -> Result<(), String> {
        self.active_calls.push(f.name.clone());

        // Setup parameters as inputs
        for param in &f.params {
            let param_wire = self.new_wire(&param.name);
            // By convention, circuit parameters are treated as Private Inputs unless marked specifically.
            // Let's make them Private Inputs, and any returned value as Output.
            self.private_inputs.push(param_wire);
            self.bind(&param.name, WireBinding::Wire(param_wire));
        }

        // Lower body
        let ret_lc = self.emit_block(&f.body, items)?;

        // If there is a return expression, register it as Output wire
        if let Some(lc) = ret_lc {
            let out_wire = self.new_wire("out_ret");
            self.outputs.push(out_wire);
            // Constrain out_wire = lc
            // R1CS: (lc) * (1) = out_wire
            self.add_constraint(Constraint {
                a: lc,
                b: LinearCombination::constant(Fr::one()),
                c: LinearCombination::variable(out_wire),
                span: Some(f.span.clone()),
            });
        }

        // Soundness check (error[Z0042]): check for under-constrained hint variables
        if !self.unconstrained_hint_vars.is_empty() {
            let (_wire, (var_name, span)) = self.unconstrained_hint_vars.iter().next().unwrap();
            return Err(format!(
                "Line {}: error[Z0042]: under-constrained variable '{}' allocated in @hint block\n  hint: variable was assigned in an unconstrained block but never referenced in a linear or quadratic R1CS constraint matrix before exiting function scope",
                span.line, var_name
            ));
        }

        self.active_calls.pop();
        Ok(())
    }

    fn emit_block(&mut self, block: &Block, items: &[Item]) -> Result<Option<LinearCombination>, String> {
        self.enter_scope();
        let mut last_ret = None;
        for stmt in &block.stmts {
            if let Some(ret) = self.emit_stmt(stmt, items)? {
                last_ret = Some(ret);
            }
        }
        self.exit_scope();
        Ok(last_ret)
    }

    fn emit_stmt(&mut self, stmt: &Stmt, items: &[Item]) -> Result<Option<LinearCombination>, String> {
        match stmt {
            Stmt::Let { name, init, bounds, .. } => {
                let lc = if let Some(expr) = init {
                    let mut lc = self.emit_expr(expr, items)?;
                    lc.simplify();
                    if let Some(c) = lc.is_constant() {
                        self.const_bindings.insert(name.clone(), c.clone());
                    }
                    self.bind(name, WireBinding::Linear(lc.clone()));
                    lc
                } else {
                    // Uninitialized variable: allocate a raw witness wire
                    let wire = self.new_wire(name);
                    let lc = LinearCombination::variable(wire);
                    self.bind(name, WireBinding::Wire(wire));
                    lc
                };

                if let Some(bounds_attr) = bounds {
                    // Let's get the max value as a constant u64 or BigUint
                    let max_lc = self.emit_expr(&bounds_attr.max, items)?;
                    if let Some(max_fr) = max_lc.is_constant() {
                        let bit_len = max_fr.bit_len();
                        
                        // We decompose lc into bit_len bits:
                        // lc = sum_{i=0}^{bit_len-1} b_i * 2^i
                        // and for each b_i, b_i * b_i = b_i
                        let mut sum_lc = LinearCombination::zero();
                        let mut pow2 = Fr::one();
                        for i in 0..bit_len {
                            let bit_var = self.new_wire(&format!("{}_bit_{}", name, i));
                            let bit_lc = LinearCombination::variable(bit_var);
                            // constraint: bit_var * bit_var = bit_var
                            self.constraints.push(Constraint {
                                a: bit_lc.clone(),
                                b: bit_lc.clone(),
                                c: bit_lc.clone(),
                                span: Some(bounds_attr.span.clone()),
                            });
                            
                            sum_lc.add_term(bit_var, pow2);
                            pow2 = pow2.double();
                        }
                        
                        // Constrain: sum_lc = lc
                        self.constraints.push(Constraint {
                            a: sum_lc,
                            b: LinearCombination::constant(Fr::one()),
                            c: lc.clone(),
                            span: Some(bounds_attr.span.clone()),
                        });

                        // Also decompose (max_val - lc) into bit_len bits to ensure lc <= max_val!
                        let mut diff_lc = LinearCombination::constant(max_fr);
                        diff_lc.add_linear(&lc, Fr::from_u64(0).sub(&Fr::one()));
                        
                        let mut diff_sum_lc = LinearCombination::zero();
                        let mut pow2 = Fr::one();
                        for i in 0..bit_len {
                            let bit_var = self.new_wire(&format!("{}_diff_bit_{}", name, i));
                            let bit_lc = LinearCombination::variable(bit_var);
                            // constraint: bit_var * bit_var = bit_var
                            self.constraints.push(Constraint {
                                a: bit_lc.clone(),
                                b: bit_lc.clone(),
                                c: bit_lc.clone(),
                                span: Some(bounds_attr.span.clone()),
                            });
                            
                            diff_sum_lc.add_term(bit_var, pow2);
                            pow2 = pow2.double();
                        }
                        
                        // Constrain: diff_sum_lc = diff_lc
                        self.constraints.push(Constraint {
                            a: diff_sum_lc,
                            b: LinearCombination::constant(Fr::one()),
                            c: diff_lc,
                            span: Some(bounds_attr.span.clone()),
                        });
                    }
                }
            }
            Stmt::Assign { target, value, span } => {
                let target_name = match target {
                    Expr::Ident(name, _) => name.clone(),
                    _ => return Err(format!("Circuit target error: Assignments to non-identifiers are not supported. Line {}", span.line)),
                };
                
                let mut optimized = false;
                if let Expr::BinaryOp { left, op, right, .. } = value {
                    // Case 1: target = target + expr
                    if let Expr::Ident(ref left_name, _) = **left {
                        if left_name == &target_name && !expr_references_var(right, &target_name) {
                            if let Some((scope_idx, binding)) = self.take_binding_from_scope(&target_name) {
                                let mut target_lc = match binding {
                                    WireBinding::Wire(w) => LinearCombination::variable(w),
                                    WireBinding::Linear(lc) => lc,
                                };
                                let right_lc = self.emit_expr(right, items)?;
                                
                                match op {
                                    BinaryOp::Add => {
                                        target_lc.add_linear(&right_lc, Fr::one());
                                        target_lc.simplify();
                                        if let Some(c) = target_lc.is_constant() {
                                            self.const_bindings.insert(target_name.clone(), c);
                                        } else {
                                            self.const_bindings.remove(&target_name);
                                        }
                                        self.bind_to_scope(scope_idx, &target_name, WireBinding::Linear(target_lc));
                                        optimized = true;
                                    }
                                    BinaryOp::Sub => {
                                        let neg_one = Fr::from_u64(0).sub(&Fr::one());
                                        target_lc.add_linear(&right_lc, neg_one);
                                        target_lc.simplify();
                                        if let Some(c) = target_lc.is_constant() {
                                            self.const_bindings.insert(target_name.clone(), c);
                                        } else {
                                            self.const_bindings.remove(&target_name);
                                        }
                                        self.bind_to_scope(scope_idx, &target_name, WireBinding::Linear(target_lc));
                                        optimized = true;
                                    }
                                    _ => {
                                        self.bind_to_scope(scope_idx, &target_name, WireBinding::Linear(target_lc));
                                    }
                                }
                            }
                        }
                    }
                    
                    // Case 2: target = expr + target
                    if !optimized {
                        if let Expr::Ident(ref right_name, _) = **right {
                            if right_name == &target_name && !expr_references_var(left, &target_name) {
                                if let Some((scope_idx, binding)) = self.take_binding_from_scope(&target_name) {
                                    let mut target_lc = match binding {
                                        WireBinding::Wire(w) => LinearCombination::variable(w),
                                        WireBinding::Linear(lc) => lc,
                                    };
                                    let left_lc = self.emit_expr(left, items)?;
                                    
                                    match op {
                                        BinaryOp::Add => {
                                            target_lc.add_linear(&left_lc, Fr::one());
                                            target_lc.simplify();
                                            if let Some(c) = target_lc.is_constant() {
                                                self.const_bindings.insert(target_name.clone(), c);
                                            } else {
                                                self.const_bindings.remove(&target_name);
                                            }
                                            self.bind_to_scope(scope_idx, &target_name, WireBinding::Linear(target_lc));
                                            optimized = true;
                                        }
                                        BinaryOp::Sub => {
                                            target_lc = target_lc.scale(Fr::from_u64(0).sub(&Fr::one()));
                                            target_lc.add_linear(&left_lc, Fr::one());
                                            target_lc.simplify();
                                            if let Some(c) = target_lc.is_constant() {
                                                self.const_bindings.insert(target_name.clone(), c);
                                            } else {
                                                self.const_bindings.remove(&target_name);
                                            }
                                            self.bind_to_scope(scope_idx, &target_name, WireBinding::Linear(target_lc));
                                            optimized = true;
                                        }
                                        _ => {
                                            self.bind_to_scope(scope_idx, &target_name, WireBinding::Linear(target_lc));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if !optimized {
                    // Validate circuit restrictions: mutable reassignment must create a multiplexed node or single-assign binding
                    let mut lc_val = self.emit_expr(value, items)?;
                    lc_val.simplify();
                    
                    // Track constant bindings
                    if let Some(c) = lc_val.is_constant() {
                        self.const_bindings.insert(target_name.clone(), c);
                    } else {
                        self.const_bindings.remove(&target_name);
                    }

                    // In ZK, re-assigning is done by binding the target to the new linear combination (SSA-style name rebinding)
                    if !self.bind_update(&target_name, WireBinding::Linear(lc_val.clone())) {
                        self.bind(&target_name, WireBinding::Linear(lc_val));
                    }
                }
            }
            Stmt::For {
                loop_var,
                start,
                end,
                step,
                body,
                span,
                ..
            } => {
                // Loop unrolling validation: loop bounds MUST be compile-time constants
                let start_lc = self.emit_expr(start, items)?;
                let end_lc = self.emit_expr(end, items)?;
                
                let start_const = start_lc.is_constant()
                    .ok_or_else(|| format!("Circuit limitation: Loop start bound must be a compile-time constant. Line {}", span.line))?;
                let end_const = end_lc.is_constant()
                    .ok_or_else(|| format!("Circuit limitation: Loop end bound must be a compile-time constant. Line {}", span.line))?;
                
                let step_val = if let Some(step_expr) = step {
                    let step_lc = self.emit_expr(step_expr, items)?;
                    step_lc.is_constant()
                        .ok_or_else(|| format!("Circuit limitation: Loop step must be a compile-time constant. Line {}", span.line))?
                } else {
                    Fr::one()
                };

                let mut current = start_const;

                let mut unroll_count: usize = 0;
                // Soft guard against a typo'd bound turning into an OOM, NOT a
                // structural limit of the lowering - nothing about R1CS or this
                // emitter breaks above it. It is overridable because real
                // circuits are routinely millions of constraints, and a
                // hardcoded 10k silently put every such circuit out of reach:
                // it is why this repo's own headline benchmark (1,000,000
                // constraints) could not be reproduced. Raise it deliberately
                // and watch memory - see docs/heavy_circuit_speed_test.md for
                // measured cost per constraint.
                let max_unroll_limit: usize = std::env::var("Y_ZK_MAX_UNROLL")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(10_000);

                while current < end_const {
                    unroll_count += 1;
                    if unroll_count > max_unroll_limit {
                        return Err(format!(
                            "Circuit unroll error [Z0099]: Loop iteration count exceeded safety unrolling threshold (max {} iterations). Line {}",
                            max_unroll_limit, span.line
                        ));
                    }

                    self.enter_scope();
                    self.const_bindings.insert(loop_var.clone(), current.clone());
                    self.bind(loop_var, WireBinding::Linear(LinearCombination::constant(current.clone())));

                    // Inline loop body
                    for body_stmt in &body.stmts {
                        self.emit_stmt(body_stmt, items)?;
                    }

                    self.exit_scope();
                    current = current.add(&step_val);
                }
            }
            Stmt::If {
                condition,
                then_block,
                else_block,
                span,
                ..
            } => {
                // Compile-time multiplexed assignment or static if evaluation
                let cond_lc = self.emit_expr(condition, items)?;
                
                if let Some(c) = cond_lc.is_constant() {
                    // Static branch pruning: evaluate only the active branch
                    if !c.is_zero() {
                        return self.emit_block(then_block, items);
                    } else if let Some(eb) = else_block {
                        return self.emit_block(eb, items);
                    }
                } else {
                    // Dynamic conditional execution: both branches must be evaluated and output wires merged using selectors
                    
                    // 1. Clone the current scopes and const_bindings for branch isolation
                    let mut then_scopes = self.scopes.clone();
                    let mut else_scopes = self.scopes.clone();
                    let pre_const = self.const_bindings.clone();

                    // Evaluate then branch
                    let original_scopes = std::mem::replace(&mut self.scopes, then_scopes);
                    self.const_bindings = pre_const.clone();
                    let then_ret = self.emit_block(then_block, items)?;
                    then_scopes = std::mem::replace(&mut self.scopes, original_scopes);
                    let then_const = self.const_bindings.clone();

                    // Evaluate else branch
                    let mut else_ret = None;
                    let mut else_const = pre_const.clone();
                    if let Some(eb) = else_block {
                        let original_scopes = std::mem::replace(&mut self.scopes, else_scopes);
                        self.const_bindings = pre_const.clone();
                        else_ret = self.emit_block(eb, items)?;
                        else_scopes = std::mem::replace(&mut self.scopes, original_scopes);
                        else_const = self.const_bindings.clone();
                    }

                    // Merge const bindings: only keep those that are identical in both branches
                    let mut merged_const = HashMap::new();
                    for (k, v1) in then_const {
                        if let Some(v2) = else_const.get(&k) {
                            if v1 == *v2 {
                                merged_const.insert(k, v1);
                            }
                        }
                    }
                    self.const_bindings = merged_const;

                    // Compare and merge the mutated scopes
                    for i in 0..self.scopes.len() {
                        let then_map = &then_scopes[i];
                        let else_map = &else_scopes[i];

                        // Collect all keys in either map at this scope level
                        let mut all_keys = HashSet::new();
                        for k in then_map.keys() {
                            all_keys.insert(k.clone());
                        }
                        for k in else_map.keys() {
                            all_keys.insert(k.clone());
                        }

                        for var in all_keys {
                            let then_val = then_map.get(&var).cloned();
                            let else_val = else_map.get(&var).cloned();

                            if then_val != else_val {
                                let then_lc = match then_val {
                                    Some(WireBinding::Wire(w)) => LinearCombination::variable(w),
                                    Some(WireBinding::Linear(lc)) => lc,
                                    None => self.lookup_in_scopes(&then_scopes, i, &var)
                                        .unwrap_or_else(LinearCombination::zero),
                                };

                                let else_lc = match else_val {
                                    Some(WireBinding::Wire(w)) => LinearCombination::variable(w),
                                    Some(WireBinding::Linear(lc)) => lc,
                                    None => self.lookup_in_scopes(&else_scopes, i, &var)
                                        .unwrap_or_else(LinearCombination::zero),
                                };

                                // Merged value wire:
                                let merged_wire = self.new_wire(&format!("{}_mux", var));
                                let merged_lc = LinearCombination::variable(merged_wire);

                                // Constraints: (cond_lc) * (then_lc - else_lc) = merged_lc - else_lc
                                let mut b_term = then_lc.clone();
                                b_term.add_linear(&else_lc, Fr::from_u64(0).sub(&Fr::one()));

                                let mut c_term = merged_lc.clone();
                                c_term.add_linear(&else_lc, Fr::from_u64(0).sub(&Fr::one()));

                                self.constraints.push(Constraint {
                                    a: cond_lc.clone(),
                                    b: b_term,
                                    c: c_term,
                                    span: Some(span.clone()),
                                });

                                // Update the actual scope level i with the multiplexed wire binding!
                                self.scopes[i].insert(var.clone(), WireBinding::Wire(merged_wire));
                            }
                        }
                    }

                    // Return values multiplexing
                    if then_ret.is_some() || else_ret.is_some() {
                        let tr = then_ret.unwrap_or_else(LinearCombination::zero);
                        let er = else_ret.unwrap_or_else(LinearCombination::zero);

                        let merged_ret_wire = self.new_wire("ret_mux");
                        let merged_ret_lc = LinearCombination::variable(merged_ret_wire);

                        let mut b_term = tr;
                        b_term.add_linear(&er, Fr::from_u64(0).sub(&Fr::one()));

                        let mut c_term = merged_ret_lc.clone();
                        c_term.add_linear(&er, Fr::from_u64(0).sub(&Fr::one()));

                        self.constraints.push(Constraint {
                            a: cond_lc,
                            b: b_term,
                            c: c_term,
                            span: Some(span.clone()),
                        });

                        return Ok(Some(merged_ret_lc));
                    }
                }
            }
            Stmt::Return(expr_opt, _span) => {
                if let Some(expr) = expr_opt {
                    let lc = self.emit_expr(expr, items)?;
                    return Ok(Some(lc));
                }
            }
            Stmt::Expr(expr) => {
                self.emit_expr(expr, items)?;
            }
            Stmt::While { condition, body, max_iterations, span, .. } => {
                let max_iters = match max_iterations {
                    Some(n) if *n > 0 => *n,
                    _ => {
                        return Err(format!(
                            "Line {}: error[Z0010]: dynamic 'while' loop prohibited in ZK circuit mode\n  hint: annotate loop with '@max_iterations(N)' where N is a compile-time constant integer",
                            span.line
                        ));
                    }
                };

                // 1. Initial condition & Active Mask initialization
                let initial_cond_lc = self.emit_expr(condition, items)?;

                // Fast Path: Compile-Time Static Loop Condition Optimization
                // If condition is a compile-time constant (e.g. static index bounds i < N),
                // execute iterations directly without emitting active-mask wires or SSA phi-nodes.
                if let Some(const_val) = initial_cond_lc.is_constant() {
                    if const_val.is_zero() {
                        return Ok(None); // Statically false condition, zero iterations
                    }
                    
                    let mut is_static_throughout = true;
                    for _iter in 0..max_iters {
                        self.emit_block(body, items)?;
                        let next_cond_lc = self.emit_expr(condition, items)?;
                        if let Some(next_c) = next_cond_lc.is_constant() {
                            if next_c.is_zero() {
                                break;
                            }
                        } else {
                            is_static_throughout = false;
                            break;
                        }
                    }

                    if is_static_throughout {
                        return Ok(None);
                    }
                }

                // Fallback Path: Dynamic Witness Loop Condition with Active Mask SSA Phi-Nodes
                let mut current_active_wire = self.allocate_var_from_lc(&initial_cond_lc);
                let neg_one = Fr::from_u64(0).sub(&Fr::one());

                // 2. Unrolled Iteration Processing (i = 0 .. max_iters)
                for _iter in 0..max_iters {
                    let scope_before = self.scopes.clone();

                    // Execute loop body in isolated scope
                    self.emit_block(body, items)?;

                    // Re-evaluate condition at end of body for next iteration mask
                    let next_cond_lc = self.emit_expr(condition, items)?;
                    let next_cond_wire = self.allocate_var_from_lc(&next_cond_lc);

                    let next_active_wire = self.new_wire("active_mask");
                    // active_{i+1} = active_i * cond_{i+1}
                    self.constraints.push(Constraint {
                        a: LinearCombination::variable(current_active_wire),
                        b: LinearCombination::variable(next_cond_wire),
                        c: LinearCombination::variable(next_active_wire),
                        span: Some(span.clone()),
                    });

                    // SSA Phi-Node Multiplexing across scope_before and self.scopes
                    for i in 0..self.scopes.len() {
                        let before_map = &scope_before[i];
                        let body_map = &self.scopes[i];

                        let mut updates = Vec::new();

                        for (var, body_val) in body_map.iter() {
                            let before_val = before_map.get(var);

                            if before_val != Some(body_val) {
                                let before_lc = match before_val {
                                    Some(WireBinding::Wire(w)) => LinearCombination::variable(*w),
                                    Some(WireBinding::Linear(lc)) => lc.clone(),
                                    None => self.lookup_in_scopes(&scope_before, i, var).unwrap_or_else(LinearCombination::zero),
                                };

                                let body_lc = match body_val {
                                    WireBinding::Wire(w) => LinearCombination::variable(*w),
                                    WireBinding::Linear(lc) => lc.clone(),
                                };

                                updates.push((var.clone(), before_lc, body_lc));
                            }
                        }

                        for (var, before_lc, body_lc) in updates {
                            let mux_wire = self.new_wire("while_mux");
                            let mux_lc = LinearCombination::variable(mux_wire);

                            let mut b_term = body_lc;
                            b_term.add_linear(&before_lc, neg_one.clone());

                            let mut c_term = mux_lc;
                            c_term.add_linear(&before_lc, neg_one.clone());

                            self.constraints.push(Constraint {
                                a: LinearCombination::variable(current_active_wire),
                                b: b_term,
                                c: c_term,
                                span: Some(span.clone()),
                            });

                            self.scopes[i].insert(var, WireBinding::Wire(mux_wire));
                        }
                    }

                    current_active_wire = next_active_wire;
                }
            }
            Stmt::HintBlock { outputs, body, span } => {
                // 1. Enter an isolated child scope for hint evaluation to prevent internal variable leakage
                self.enter_scope();
                let _ = self.emit_block(body, items)?;
                self.exit_scope();

                // 2. Only allocate wire indices and bind variables explicitly listed in outputs into the outer scope
                for out_name in outputs {
                    let wire = self.new_wire(out_name);
                    self.bind(out_name, WireBinding::Wire(wire));
                    self.unconstrained_hint_vars.insert(wire, (out_name.clone(), span.clone()));
                }
            }
            Stmt::Break { span } => {
                return Err(format!("Circuit target error: 'break' statements are forbidden in ZK circuits. Line {}", span.line));
            }
            Stmt::Match { span, .. } => {
                return Err(format!("Circuit target error: Pattern matching is currently not supported in ZK circuits. Line {}", span.line));
            }
            _ => {}
        }
        Ok(None)
    }

    fn emit_expr(&mut self, expr: &Expr, items: &[Item]) -> Result<LinearCombination, String> {
        match expr {
            Expr::IntLit(val, _) => {
                Ok(LinearCombination::constant(Fr::from_u64(*val as u64)))
            }
            Expr::BoolLit(val, _) => {
                let v = if *val { Fr::one() } else { Fr::zero() };
                Ok(LinearCombination::constant(v))
            }
            Expr::Ident(name, span) => {
                if let Some(c) = self.lookup_const(name) {
                    return Ok(LinearCombination::constant(c));
                }
                match self.lookup(name) {
                    Some(WireBinding::Wire(id)) => Ok(LinearCombination::variable(id)),
                    Some(WireBinding::Linear(lc)) => Ok(lc),
                    None => Err(format!("Undefined variable {} in circuit expression. Line {}", name, span.line)),
                }
            }
            Expr::BinaryOp { left, op, right, span } => {
                let left_lc = self.emit_expr(left, items)?;
                let right_lc = self.emit_expr(right, items)?;

                match op {
                    BinaryOp::Add => {
                        let mut res = left_lc;
                        res.add_linear(&right_lc, Fr::one());
                        res.simplify();
                        Ok(res)
                    }
                    BinaryOp::Sub => {
                        let mut res = left_lc;
                        let neg_one = Fr::from_u64(0).sub(&Fr::one());
                        res.add_linear(&right_lc, neg_one);
                        res.simplify();
                        Ok(res)
                    }
                    BinaryOp::Mul => {
                        // Optimizations:
                        // Constant * Constant -> Constant
                        // Constant * LC -> LC scaled
                        if let Some(lc) = left_lc.is_constant() {
                            return Ok(right_lc.scale(lc));
                        }
                        if let Some(rc) = right_lc.is_constant() {
                            return Ok(left_lc.scale(rc));
                        }

                        // Otherwise, non-linear multiplication requires a new constraint wire
                        let out_wire = self.new_wire("mul_tmp");
                        self.add_constraint(Constraint {
                            a: left_lc,
                            b: right_lc,
                            c: LinearCombination::variable(out_wire),
                            span: Some(span.clone()),
                        });
                        Ok(LinearCombination::variable(out_wire))
                    }
                    BinaryOp::Div => {
                        // INTEGER division, matching the `I32` the source says.
                        //
                        // This used to be FIELD division: `out * right = left`,
                        // i.e. `left * right^-1 mod p`. For `7 / 2` that is
                        // `(p+7)/2`, a 254-bit number - not 3. The two agree
                        // only when the divisor divides the dividend exactly,
                        // so every other case was silently wrong, and wrong in
                        // a way that still produces a valid proof. It also has
                        // to match `%`: with field division,
                        // `(a/b)*b + a%b == a` fails.
                        if let (Some(l), Some(r)) = (lc_u64(&left_lc), lc_u64(&right_lc)) {
                            if r == 0 {
                                return Err(format!(
                                    "Circuit target error: division by zero. Line {}",
                                    span.line
                                ));
                            }
                            return Ok(LinearCombination::constant(Fr::from_u64(l / r)));
                        }
                        let (quot, _) =
                            self.emit_int_div_mod(&left_lc, &right_lc, ZK_COMPARISON_BITS, span);
                        Ok(quot)
                    }
                    BinaryOp::Eq => {
                        // Equality: left == right -> outputs 1 if equal, 0 if not
                        // Constraint-based check:
                        // Let d = left - right
                        // We introduce a helper wire inv_d.
                        // Constrain:
                        // 1) d * (1 - eq) = 0
                        // 2) d * inv_d = eq
                        let mut d = left_lc;
                        d.add_linear(&right_lc, Fr::from_u64(0).sub(&Fr::one()));

                        if let Some(dc) = d.is_constant() {
                            let eq_val = if dc.is_zero() { Fr::one() } else { Fr::zero() };
                            return Ok(LinearCombination::constant(eq_val));
                        }

                        let eq_wire = self.new_wire("eq_tmp");
                        let inv_d_wire = self.new_wire("inv_d_tmp");
                        // Both constraints below carry two unknowns, so the
                        // witness pass cannot back-propagate either wire -
                        // record how to compute them directly. Without this,
                        // every circuit using `==`/`!=` (and so every
                        // comparison) was unwitnessable and therefore
                        // unprovable.
                        self.witness_recipes
                            .insert(eq_wire, WitnessOp::IsZeroLc(d.clone()));
                        self.witness_recipes
                            .insert(inv_d_wire, WitnessOp::InvOrZeroLc(d.clone()));

                        // Constraint 1: d * eq = 0
                        self.add_constraint(Constraint {
                            a: d.clone(),
                            b: LinearCombination::variable(eq_wire),
                            c: LinearCombination::zero(),
                            span: Some(span.clone()),
                        });

                        // Constraint 2: d * inv_d = 1 - eq
                        let mut c_term = LinearCombination::constant(Fr::one());
                        c_term.add_term(eq_wire, Fr::from_u64(0).sub(&Fr::one()));

                        self.add_constraint(Constraint {
                            a: d,
                            b: LinearCombination::variable(inv_d_wire),
                            c: c_term,
                            span: Some(span.clone()),
                        });

                        Ok(LinearCombination::variable(eq_wire))
                    }
                    BinaryOp::NotEq => {
                        // Not Equal: (left == right) == 0
                        let eq_lc = self.emit_expr(&Expr::BinaryOp {
                            left: left.clone(),
                            op: BinaryOp::Eq,
                            right: right.clone(),
                            span: span.clone(),
                        }, items)?;

                        // return 1 - eq
                        let mut res = LinearCombination::constant(Fr::one());
                        res.add_linear(&eq_lc, Fr::from_u64(0).sub(&Fr::one()));
                        Ok(res)
                    }
                    BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                        if let (Some(lc), Some(rc)) = (left_lc.is_constant(), right_lc.is_constant()) {
                            let is_true = match op {
                                BinaryOp::Lt => lc < rc,
                                BinaryOp::Le => lc <= rc,
                                BinaryOp::Gt => lc > rc,
                                BinaryOp::Ge => lc >= rc,
                                _ => unreachable!(),
                            };
                            let val = if is_true { Fr::one() } else { Fr::zero() };
                            Ok(LinearCombination::constant(val))
                        } else {
                            // Real ordering comparison via bit decomposition.
                            //
                            // This arm previously re-emitted the expression as
                            // `BinaryOp::NotEq` and returned that, so `x < y`
                            // lowered to `x != y`. That is not an
                            // under-constrained comparison, it is a DIFFERENT
                            // FUNCTION: `5 <= 5` came out false, and `5 > 3`
                            // and `3 > 5` were both true. Groth16 will prove a
                            // wrong statement just as happily as a right one,
                            // so this was a soundness hole, not a precision
                            // issue. All four operators emitted an identical 3
                            // constraints, which is the tell - see
                            // `emit_less_than` for what the real cost is.
                            let n = ZK_COMPARISON_BITS;
                            let lt = match op {
                                // a < b, and a > b is just b < a.
                                BinaryOp::Lt => self.emit_less_than(&left_lc, &right_lc, n, span),
                                BinaryOp::Gt => self.emit_less_than(&right_lc, &left_lc, n, span),
                                // a <= b  <=>  !(b < a);  a >= b  <=>  !(a < b)
                                BinaryOp::Le => {
                                    let gt = self.emit_less_than(&right_lc, &left_lc, n, span);
                                    let mut r = LinearCombination::constant(Fr::one());
                                    r.add_linear(&gt, Fr::from_u64(0).sub(&Fr::one()));
                                    r
                                }
                                BinaryOp::Ge => {
                                    let lt = self.emit_less_than(&left_lc, &right_lc, n, span);
                                    let mut r = LinearCombination::constant(Fr::one());
                                    r.add_linear(&lt, Fr::from_u64(0).sub(&Fr::one()));
                                    r
                                }
                                _ => unreachable!(),
                            };
                            Ok(lt)
                        }
                    }
                    BinaryOp::Mod => {
                        if let (Some(l), Some(r)) = (lc_u64(&left_lc), lc_u64(&right_lc)) {
                            if r == 0 {
                                return Err(format!(
                                    "Circuit target error: `% 0` is not satisfiable. Line {}",
                                    span.line
                                ));
                            }
                            return Ok(LinearCombination::constant(Fr::from_u64(l % r)));
                        }
                        let (_, rem) =
                            self.emit_int_div_mod(&left_lc, &right_lc, ZK_COMPARISON_BITS, span);
                        Ok(rem)
                    }
                    BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => {
                        if let (Some(l), Some(r)) = (lc_u64(&left_lc), lc_u64(&right_lc)) {
                            let v = match op {
                                BinaryOp::BitAnd => l & r,
                                BinaryOp::BitOr => l | r,
                                _ => l ^ r,
                            };
                            return Ok(LinearCombination::constant(Fr::from_u64(v)));
                        }
                        Ok(self.emit_bitwise(&left_lc, &right_lc, op, ZK_COMPARISON_BITS, span))
                    }
                    BinaryOp::Shl | BinaryOp::Shr => {
                        // A variable shift amount is a multiplexer over every
                        // possible amount - a different gadget at a different
                        // price. Refusing is better than quietly emitting one.
                        let amount = lc_u64(&right_lc).ok_or_else(|| {
                            format!(
                                "Circuit target error: the shift amount in `{:?}` must be a \
                                 compile-time constant in ZK circuits; a variable shift needs a \
                                 multiplexer over all {} possible amounts. Line {}",
                                op, ZK_COMPARISON_BITS, span.line
                            )
                        })?;
                        let left = matches!(op, BinaryOp::Shl);
                        if let Some(l) = lc_u64(&left_lc) {
                            let n = ZK_COMPARISON_BITS as u64;
                            let mask = (1u64 << n) - 1;
                            let v = if amount >= n {
                                0
                            } else if left {
                                (l << amount) & mask
                            } else {
                                (l & mask) >> amount
                            };
                            return Ok(LinearCombination::constant(Fr::from_u64(v)));
                        }
                        Ok(self.emit_shift(&left_lc, amount, left, ZK_COMPARISON_BITS, span))
                    }
                    BinaryOp::And | BinaryOp::Or => {
                        // Logical operators. Y's comparisons already return 0/1,
                        // but an arbitrary I32 does not, so both operands are
                        // constrained boolean - `5 && 3` must fail to prove
                        // rather than produce whatever the field says.
                        self.constrain_boolean(&left_lc, span);
                        self.constrain_boolean(&right_lc, span);
                        let and = self.emit_mul_lc(&left_lc, &right_lc, "logic_and");
                        if matches!(op, BinaryOp::And) {
                            return Ok(and);
                        }
                        // a || b  =  a + b - ab
                        let mut out = left_lc;
                        out.add_linear(&right_lc, Fr::one());
                        out.add_linear(&and, Fr::from_u64(0).sub(&Fr::one()));
                        out.simplify();
                        Ok(out)
                    }
                }
            }
            Expr::Call { func, args, span } => {
                let func_name = match &**func {
                    Expr::Ident(name, _) => name.clone(),
                    _ => return Err(format!("Invalid function call in circuit. Line {}", span.line)),
                };

                if func_name == "poseidon_hash" {
                    return self.emit_poseidon(args, items);
                }

                // Evaluate argument linear combinations in current active scope
                let mut arg_lcs = Vec::new();
                for arg in args {
                    arg_lcs.push(self.emit_expr(arg, items)?);
                }

                // Validate ZK Recursion Restrictions (Static vs Dynamic)
                if self.active_calls.len() > 256 {
                    return Err(format!(
                        "Line {}: error[Z0011]: maximum recursion depth exceeded during ZK circuit monomorphization\n  hint: recursion depth must be statically bounded at compile time",
                        span.line
                    ));
                }

                if self.active_calls.contains(&func_name) {
                    // Check if all recursive call arguments are non-constant (witness-dependent)
                    let any_const_arg = arg_lcs.iter().any(|lc| lc.is_constant().is_some());
                    if !any_const_arg {
                        return Err(format!(
                            "Line {}: error[Z0012]: dynamic witness-dependent recursion is prohibited in R1CS targets",
                            span.line
                        ));
                    }
                }

                // Lookup function declaration
                let mut target_decl = None;
                for item in items {
                    if let Item::Func(f) = item {
                        if f.name == func_name {
                            target_decl = Some(f);
                            break;
                        }
                    }
                }

                let f = target_decl.ok_or_else(|| format!("Undefined function {} called. Line {}", func_name, span.line))?;

                // Inline function execution: evaluate arguments and bind to parameters
                self.enter_scope();
                self.active_calls.push(func_name.clone());

                for (param, arg_lc) in f.params.iter().zip(arg_lcs) {
                    self.bind(&param.name, WireBinding::Linear(arg_lc));
                }

                let ret_lc = self.emit_block(&f.body, items)?;

                self.active_calls.pop();
                self.exit_scope();

                ret_lc.ok_or_else(|| format!("Function {} did not return a value in circuit context. Line {}", func_name, span.line))
            }
            Expr::Index { base, index, span } => {
                // Validate circuit restrictions: No dynamic pointer arithmetic / dynamic indexing
                // Array elements must be indexed with compile-time constants
                let index_lc = self.emit_expr(index, items)?;
                let index_const = index_lc.is_constant()
                    .ok_or_else(|| format!("Circuit limitation violated: Dynamic pointer arithmetic or dynamic array indexing is forbidden in ZK circuits. Index must be a compile-time constant. Line {}", span.line))?;
                
                // Let's resolve the array lookup
                let base_name = match &**base {
                    Expr::Ident(name, _) => name,
                    _ => return Err(format!("Forbidden index target in ZK backend. Line {}", span.line)),
                };

                // Look up in scopes for bindings
                let index_val = index_const.to_decimal_string();
                let indexed_name = format!("{}_{}", base_name, index_val);
                
                if let Some(c) = self.lookup_const(&indexed_name) {
                    return Ok(LinearCombination::constant(c));
                }
                match self.lookup(&indexed_name) {
                    Some(WireBinding::Wire(id)) => Ok(LinearCombination::variable(id)),
                    Some(WireBinding::Linear(lc)) => Ok(lc),
                    None => Err(format!("Undefined array element {}[{}] in circuit. Line {}", base_name, index_val, span.line)),
                }
            }
            _ => Err(format!("Circuit target error: Expression {:?} is unsupported in ZK backends. Line {}", expr, expr.span().line)),
        }
    }

    /// One R1CS constraint `a * b = w` for a fresh wire `w`.
    ///
    /// R1CS allows `a` and `b` to be arbitrary linear combinations, which is
    /// what makes Poseidon cheap - `(x + C)^2` needs no wire for `x + C`.
    ///
    /// This used to ALSO store `WitnessOp::MulLc(a.clone(), b.clone())` in
    /// `witness_recipes`, because `build_witness_ir`'s reconstruction scan once
    /// recognised only single-term sides. It has a second-chance scan for
    /// exactly this shape now (`lc_by_output`), so the recipe was a verbatim
    /// second copy of `a` and `b` - measured at **1096 bytes per constraint**
    /// on a Poseidon circuit, against 1135 for the constraints themselves. It
    /// nearly doubled the emitter's live memory, on every compile, including the
    /// ones that never generate a witness.
    ///
    /// The reconstruction is exact for this shape and its guard is satisfied by
    /// construction: `wire` is freshly allocated, so every wire `a` or `b`
    /// mentions is necessarily smaller, which is what `lc_by_output`'s
    /// `all(|(w, _)| *w < out)` check requires. `optimize_circuit` only ever
    /// rewrites a wire to a smaller one, so that stays true afterwards.
    ///
    /// Memory is the binding constraint on the circuit sizes this compiler
    /// exists to reach - not storing the same linear combination twice is worth
    /// more than saving a scan that only runs when a witness is requested.
    fn emit_mul_lc(
        &mut self,
        a: &LinearCombination,
        b: &LinearCombination,
        name: &str,
    ) -> LinearCombination {
        let wire = self.new_wire(name);
        self.constraints.push(Constraint {
            a: a.clone(),
            b: b.clone(),
            c: LinearCombination::variable(wire),
            span: None,
        });
        LinearCombination::variable(wire)
    }

    /// Poseidon's S-box, `x^5`, in 3 constraints.
    fn emit_poseidon_sbox(&mut self, x: &LinearCombination) -> LinearCombination {
        // A fully constant lane (the initial capacity element is 0, and literal
        // arguments stay constant through the first rounds) folds instead of
        // burning constraints. Same value either way.
        if let Some(c) = x.is_constant() {
            let c2 = c.mul(&c);
            let c4 = c2.mul(&c2);
            return LinearCombination::constant(c4.mul(&c));
        }
        let x2 = self.emit_mul_lc(x, x, "pos_x2");
        let x4 = self.emit_mul_lc(&x2, &x2, "pos_x4");
        self.emit_mul_lc(&x4, x, "pos_x5")
    }

    /// circomlib-compatible Poseidon over BN254.
    ///
    /// This function used to be a Poseidon-shaped construction rather than
    /// Poseidon: the round constants came from an LCG seeded with `0x12345678`,
    /// only 60 were generated where t=3 needs 195, and the remaining 135 all
    /// fell through to a hardcoded `123456789`. The MDS was a hand-rolled 3x3
    /// Cauchy matrix. The result was a permutation, and it hashed
    /// deterministically, so nothing downstream complained - but it agreed with
    /// no other implementation on earth, and repeated round constants are
    /// precisely what Poseidon's security argument against algebraic attacks
    /// assumes you do not have. A Merkle proof built by circomlib could not be
    /// verified here, nor the reverse.
    ///
    /// Now it transcribes circomlib's parameters (see
    /// `zk_poseidon_constants.rs`) and follows the same optimized round
    /// structure, so the digests are equal. `poseidon_matches_circomlib` pins
    /// that against the published test vector.
    ///
    /// Costs 243 constraints: 81 S-boxes (`t*R_F + R_P = 3*8 + 57`) at 3 each.
    /// Every mix is a linear combination and therefore free.
    fn emit_poseidon(&mut self, args: &[Expr], items: &[Item]) -> Result<LinearCombination, String> {
        const T: usize = 3;

        // Fail closed on anything we do not have real parameters for. The
        // alternative - extending the constant table by generating filler, as
        // the previous implementation effectively did - produces a hash that is
        // wrong in a way no test downstream can see.
        if args.len() != T - 1 {
            return Err(format!(
                "Circuit target error: poseidon_hash currently supports exactly {} inputs (t={}), \
                 got {}. Y ships circomlib's t=3 parameters, which is the arity Merkle trees and \
                 commitments use; other arities would need their constant tables transcribed too, \
                 and inventing them would produce a hash incompatible with every other tool.",
                T - 1,
                T,
                args.len()
            ));
        }
        if self.active_field != ScalarField::Bn254 {
            return Err(format!(
                "Circuit target error: poseidon_hash is parameterised for BN254; the active field \
                 is {:?}. Poseidon's round constants and MDS are field-specific - reusing BN254's \
                 over another field is not a different-but-valid hash, it is an unanalysed one.",
                self.active_field
            ));
        }

        let params = poseidon_t3_params()?;
        let PoseidonT3Params { c, s_sparse, m, p } = &*params;

        let r_f = POSEIDON_T3_ROUNDS_F;
        let r_p = POSEIDON_T3_ROUNDS_P;
        let half_f = r_f / 2;

        // `out[i] = sum_j mat[j][i] * in[j]` - note the transposed index, which
        // is circomlib's convention in `Mix`. Getting this backwards yields a
        // valid-looking hash that differs from circomlib's.
        let mix = |state: &[LinearCombination], mat: &[[Fr; T]; T]| -> Vec<LinearCombination> {
            let mut out = vec![LinearCombination::zero(); T];
            for i in 0..T {
                for (j, s) in state.iter().enumerate().take(T) {
                    out[i].add_linear(s, mat[j][i]);
                }
                out[i].simplify();
            }
            out
        };

        // state = [capacity 0, inputs...]
        let mut state = Vec::with_capacity(T);
        state.push(LinearCombination::constant(Fr::zero()));
        for arg in args {
            state.push(self.emit_expr(arg, items)?);
        }

        // Ark(t, C, 0)
        for (j, s) in state.iter_mut().enumerate() {
            s.add_constant(c[j]);
        }

        // First half of the full rounds, all but the last.
        for r in 0..half_f - 1 {
            let sig: Vec<_> = state.iter().map(|x| self.emit_poseidon_sbox(&x.clone())).collect::<Vec<_>>();
            let mut arked = sig;
            for (j, s) in arked.iter_mut().enumerate() {
                s.add_constant(c[(r + 1) * T + j]);
            }
            state = mix(&arked, m);
        }

        // Last full round of the first half mixes with P, the change of basis
        // into the sparse partial-round representation.
        {
            let sig: Vec<_> = state.iter().map(|x| self.emit_poseidon_sbox(&x.clone())).collect::<Vec<_>>();
            let mut arked = sig;
            for (j, s) in arked.iter_mut().enumerate() {
                s.add_constant(c[half_f * T + j]);
            }
            state = mix(&arked, p);
        }

        // Partial rounds: S-box on lane 0 only, mixed by the sparse matrices.
        let width = T * 2 - 1;
        for r in 0..r_p {
            let mut inp = state.clone();
            inp[0] = self.emit_poseidon_sbox(&state[0]);
            inp[0].add_constant(c[(half_f + 1) * T + r]);

            let mut out = vec![LinearCombination::zero(); T];
            for (i, s) in inp.iter().enumerate().take(T) {
                out[0].add_linear(s, s_sparse[width * r + i]);
            }
            out[0].simplify();
            for j in 1..T {
                out[j] = inp[j].clone();
                out[j].add_linear(&inp[0], s_sparse[width * r + T + j - 1]);
                out[j].simplify();
            }
            state = out;
        }

        // Second half of the full rounds, all but the last.
        for r in 0..half_f - 1 {
            let sig: Vec<_> = state.iter().map(|x| self.emit_poseidon_sbox(&x.clone())).collect::<Vec<_>>();
            let mut arked = sig;
            for (j, s) in arked.iter_mut().enumerate() {
                s.add_constant(c[(half_f + 1) * T + r_p + r * T + j]);
            }
            state = mix(&arked, m);
        }

        // Final round has no Ark; MixLast takes column 0 only.
        let sig: Vec<_> = state.iter().map(|x| self.emit_poseidon_sbox(&x.clone())).collect::<Vec<_>>();
        let mut out = LinearCombination::zero();
        for (j, s) in sig.iter().enumerate().take(T) {
            out.add_linear(s, m[j][0]);
        }
        out.simplify();
        Ok(out)
    }
    fn allocate_var_from_lc(&mut self, lc: &LinearCombination) -> usize {
        if lc.terms.len() == 1 && lc.terms[0].0 != 0 && lc.terms[0].1 == Fr::one() {
            return lc.terms[0].0;
        }
        let var_id = self.new_wire("pos_tmp");
        let var_lc = LinearCombination::variable(var_id);
        self.constraints.push(Constraint {
            a: lc.clone(),
            b: LinearCombination::constant(Fr::one()),
            c: var_lc,
            span: None,
        });
        var_id
    }

    // ────────────────────────────────────────────────────────
    // 4. Circuit Optimizations
    // ────────────────────────────────────────────────────────

    fn optimize_circuit(&mut self) {
        // Runs constraint-reduction pass:
        // - Merges terms, normalizes, and flattens
        for c in &mut self.constraints {
            c.a.simplify();
            c.b.simplify();
            c.c.simplify();
        }

        // Membership sets, not the `Vec::contains` linear scans this used to do.
        // The predicate runs once per constraint, so a circuit with many inputs
        // paid O(constraints * inputs) for a question that is O(1).
        let boundary: HashSet<usize> = self
            .public_inputs
            .iter()
            .chain(self.private_inputs.iter())
            .chain(self.outputs.iter())
            .copied()
            .collect();

        let mut iteration = 0;
        loop {
            // Two independent reductions, run to a shared fixpoint because each
            // exposes work for the other: substituting a linear constraint away
            // rewrites the `A` and `B` of its consumers, which is what makes two
            // products compare equal; and merging two products renames a wire,
            // which can make a linear constraint eliminable.
            let t = std::time::Instant::now();
            let substituted = self.substitute_linear_constraints(&boundary);
            let t_sub = t.elapsed();
            let after_sub = self.constraints.len();
            let t = std::time::Instant::now();
            let deduplicated = if cse_enabled() {
                self.dedup_identical_products(&boundary)
            } else {
                false
            };
            if std::env::var_os("Y_ZK_TIMING").is_some() {
                eprintln!(
                    "[Y ZK TIMING]     opt round {}: linsub {:>7.3} s -> {:>8} | cse {:>7.3} s -> {:>8}",
                    iteration,
                    t_sub.as_secs_f64(),
                    after_sub,
                    t.elapsed().as_secs_f64(),
                    self.constraints.len()
                );
            }
            let _ = (substituted, deduplicated);
            if !substituted && !deduplicated {
                break;
            }
            iteration += 1;
            if iteration > 10 {
                break; // Safety limit
            }
        }

        // Once, after the fixpoint - not inside it. Both passes index arrays by
        // wire id for the duration of a round, so renumbering between rounds
        // would be renumbering under them.
        let t = std::time::Instant::now();
        let before = self.next_var_id;
        let compacted = self.compact_wires();
        if compacted && std::env::var_os("Y_ZK_TIMING").is_some() {
            eprintln!(
                "[Y ZK TIMING]     compact wires {:>7.3} s -> {} wires (was {})",
                t.elapsed().as_secs_f64(),
                self.next_var_id,
                before
            );
        }
    }

    /// Common-subexpression elimination: two constraints with the same `A` and
    /// `B` computing two different intermediate wires become one.
    ///
    /// Returns whether anything changed.
    fn dedup_identical_products(&mut self, boundary: &HashSet<usize>) -> bool {
        let mut any = false;
        let mut iteration = 0;
        loop {
            let mut replacements: HashMap<usize, usize> = HashMap::new();
            // Buckets hold constraint INDICES. They used to hold clones of `A`
            // and `B`, which is two ~1 KB `Vec` allocations per constraint to
            // duplicate data already sitting in `self.constraints` - 480,000 of
            // them on a 1000-hash Poseidon chain, purely so a hash collision
            // could be resolved.
            //
            // And the buckets themselves are gone now: an open-addressed table
            // of `u32` indices with linear probing allocates once for the whole
            // pass instead of once per distinct product, and a miss - which is
            // the overwhelmingly common case - is a single array read rather
            // than a `HashMap` lookup that has to construct an empty `Vec` on
            // insert. Collisions are still resolved by full equality, so the
            // table's only job is to keep the comparisons rare.
            let cap = (self.constraints.len() * 2).next_power_of_two().max(16);
            let mask = cap - 1;
            let mut seen: Vec<u32> = vec![u32::MAX; cap];
            let mut duplicate_indices: Vec<bool> = vec![false; self.constraints.len()];

            for (idx, c) in self.constraints.iter().enumerate() {
                // We only optimize constraints where C is a single intermediate wire: w_j
                // A wire is intermediate if it is not 0, not in public/private inputs, and not in outputs.
                if c.c.terms.len() != 1 {
                    continue;
                }
                let (wire_j, coeff) = (c.c.terms[0].0, c.c.terms[0].1);
                if coeff != Fr::one() || wire_j == 0 || boundary.contains(&wire_j) {
                    continue;
                }

                // Commutative in (A, B), so `a*b` and `b*a` land together.
                let combined_hash = hash_lc(&c.a).wrapping_add(hash_lc(&c.b));

                // `fx_mix`'s multiply leaves its entropy in the high bits, and
                // the table indexes with the low ones.
                let mut slot = ((combined_hash ^ (combined_hash >> 29)) as usize) & mask;
                let mut found_wire = None;
                loop {
                    let entry = seen[slot];
                    if entry == u32::MAX {
                        break;
                    }
                    let s = &self.constraints[entry as usize];
                    if (c.a == s.a && c.b == s.b) || (c.a == s.b && c.b == s.a) {
                        found_wire = Some(s.c.terms[0].0);
                        break;
                    }
                    slot = (slot + 1) & mask;
                }

                if let Some(wire_i) = found_wire {
                    replacements.insert(wire_j, wire_i);
                    duplicate_indices[idx] = true;
                } else {
                    seen[slot] = idx as u32;
                }
            }

            if replacements.is_empty() {
                break;
            }
            any = true;

            // Wire ids are dense, so the substitution is an array lookup, not a
            // hash. This runs once per TERM - about 7 M times per iteration on a
            // 1000-hash Poseidon chain, four iterations deep - and a `HashMap`
            // probe for what is overwhelmingly a miss was most of the pass.
            let mut subst: Vec<usize> = (0..self.next_var_id).collect();
            for (&old_w, &new_w) in &replacements {
                if old_w < subst.len() {
                    subst[old_w] = new_w;
                }
            }

            // Drop the duplicates in place rather than moving every surviving
            // constraint into a freshly grown second vector each iteration.
            let mut idx = 0;
            self.constraints.retain(|_| {
                let keep = !duplicate_indices[idx];
                idx += 1;
                keep
            });
            for c in &mut self.constraints {
                replace_wires_in_lc(&mut c.a, &subst);
                replace_wires_in_lc(&mut c.b, &subst);
                replace_wires_in_lc(&mut c.c, &subst);
            }

            // The witness recipes reference wires too, and this is not optional.
            //
            // A gadget's recipe carries a `LinearCombination` captured at EMIT
            // time (`emit_num2bits` keeps the value it decomposes,
            // `emit_int_div_mod` keeps its dividend and divisor, the is-zero
            // gadget keeps its difference). Rewriting the constraints without
            // rewriting those left every such recipe reading a wire this pass
            // had just deleted, which the forward pass then evaluates as ZERO.
            //
            // The result was not a wrong proof - it was NO proof: `let a = x*y;
            // let b = x*y; return a == b;` compiled to a satisfiable circuit
            // that `solve_r1cs_witness` could not solve, and reported only as
            // `satisfied = false`. Every gadget was affected (`<`, `<=`, `==`,
            // `/`, `%`, `&`, `|`, `^`, shifts), and only when the operand
            // happened to be a common subexpression - so the failure looked like
            // "this particular circuit is unprovable", not like a compiler bug.
            // `tests/zk_cse_gadget_wires.rs` pins it, controls included.
            self.remap_witness_recipes(&replacements, &subst);

            iteration += 1;
            if iteration > 10 {
                break; // Safety limit
            }
        }
        any
    }

    /// Rewrite every wire reference held by a witness recipe.
    ///
    /// Called from `optimize_circuit` on each fixpoint iteration, so chained
    /// replacements compose. Replacements only ever point a later wire at an
    /// earlier one, so this cannot make a recipe forward-reference.
    fn remap_witness_recipes(&mut self, reps: &HashMap<usize, usize>, subst: &[usize]) {
        for op in self.witness_recipes.values_mut() {
            remap_witness_op(op, subst);
        }

        // A recipe attached to an eliminated wire belongs to the survivor. The
        // two are provably equal - identical `A` and `B` is exactly why the pass
        // merged them - so `or_insert` keeping the survivor's own recipe when it
        // has one is not a choice between two answers.
        let moved: Vec<(usize, WitnessOp)> = reps
            .iter()
            .filter_map(|(old, new)| self.witness_recipes.get(old).map(|op| (*new, op.clone())))
            .collect();
        for (new, op) in moved {
            self.witness_recipes.entry(new).or_insert(op);
        }
    }

    /// Eliminate intermediate wires that a *linear* constraint already defines.
    ///
    /// This is circom's `--O1`/`--O2` linear substitution, and the reason it
    /// exists here is measurable: `Poseidon(2)` compiled through Y's circom
    /// front end was **765 constraints against circom's 517**, because every
    /// `out <== in` in the source survived as a constraint of its own. That is
    /// 48% more work for whatever proves the circuit afterwards, and it made a
    /// 200-hash chain slower in Y than in circom despite a back end that is
    /// otherwise several times faster.
    ///
    /// # Why it is sound
    ///
    /// A constraint of the form `k * L = c_w * w`, where `k` is a constant and
    /// `w` an intermediate wire appearing nowhere else in that constraint,
    /// *defines* `w`: it is satisfiable iff `w = k * L / c_w`. Substituting that
    /// expression for `w` everywhere and deleting the constraint yields an
    /// equisatisfiable system, because `w` is existentially quantified — it is
    /// neither an input nor an output, so no verifier ever sees it. Any solution
    /// of the reduced system extends to one of the original by computing `w`;
    /// any solution of the original restricts to one of the reduced.
    ///
    /// That argument is why the boundary check is not a nicety. Eliminating a
    /// public input or an output would change the *statement*, not just its
    /// encoding.
    ///
    /// # Why the pivot must be `C`'s single term
    ///
    /// Deleting a constraint removes it from `build_witness_ir`'s reconstruction
    /// scan as well. That scan derives a wire's value from the constraint whose
    /// `C` is that wire alone — so if the pivot is any *other* wire in the
    /// constraint, the wire in `C` can silently lose its only definition and the
    /// witness pass leaves it unsolved. Pivoting on exactly the wire the scan
    /// would have used means this pass takes over a definition rather than
    /// destroying one, and it hands that wire an explicit recipe in exchange.
    ///
    /// Returns whether anything changed.
    fn substitute_linear_constraints(&mut self, boundary: &HashSet<usize>) -> bool {
        let Some(budget) = linsub_fill_budget() else {
            return false;
        };

        // Wires a recipe refers to by `SignalId` cannot be rewritten into a
        // linear combination - there is nowhere to put one. Rather than teach
        // every such variant an impossible trick, they are excluded from
        // elimination outright: both the wire the recipe computes and every
        // wire it reads. `HintBlock` additionally computes several wires at
        // once, so taking over one of its outputs would orphan the others.
        //
        // Matched exhaustively with no `_ =>` arm, for the reason in
        // `remap_witness_op`: a new variant must be a compile error here, not a
        // wire quietly rewritten into something its recipe cannot express.
        let mut opaque: HashSet<usize> = HashSet::new();
        for (wire, op) in &self.witness_recipes {
            match op {
                WitnessOp::Const(_)
                | WitnessOp::Unknown
                | WitnessOp::LoadInput { .. }
                | WitnessOp::IsZeroLc(_)
                | WitnessOp::InvOrZeroLc(_)
                | WitnessOp::BitOfLc { .. }
                | WitnessOp::IntDivLc(..)
                | WitnessOp::IntModLc(..)
                | WitnessOp::MulLc(..)
                | WitnessOp::DivLc(..)
                | WitnessOp::MulAddLc(..) => {}
                // Holds only linear combinations, at every nesting level, so
                // substitution can rewrite it - but only if the branches really
                // are LC-only. `witness_op_is_lc_only` walks them and puts the
                // wire in `opaque` if any nested variant takes a `SignalId`,
                // which is the same guard the arms below apply at depth 0.
                WitnessOp::IfZeroLc(..) => {
                    if !witness_op_is_lc_only(op) {
                        opaque.insert(*wire);
                    }
                }
                WitnessOp::Add(a, b)
                | WitnessOp::Sub(a, b)
                | WitnessOp::Mul(a, b)
                | WitnessOp::Div(a, b)
                | WitnessOp::AssertEq(a, b) => {
                    opaque.extend([*wire, a.0, b.0]);
                }
                WitnessOp::Inv(a) => {
                    opaque.extend([*wire, a.0]);
                }
                WitnessOp::HintBlock { inputs, outputs, ops } => {
                    opaque.insert(*wire);
                    opaque.extend(inputs.iter().chain(outputs.iter()).map(|s| s.0));
                    for hop in ops {
                        match hop {
                            HintOp::NonDeterministicInv { src, dst }
                            | HintOp::AssignExpr { dst, src } => {
                                opaque.extend([src.0, dst.0]);
                            }
                            HintOp::BitDecompose { src, dst_bits } => {
                                opaque.insert(src.0);
                                opaque.extend(dst_bits.iter().map(|s| s.0));
                            }
                        }
                    }
                }
            }
        }

        // How many times each wire is mentioned, for the fill-in estimate.
        // Built on demand: a circuit with no eliminable constraint at all - Y's
        // own front end, every round after the first - should not pay a full
        // scan of every term to discover that.
        let mut occ: Vec<u32> = Vec::new();

const NONE: u32 = u32::MAX;
        let mut sub_idx: Vec<u32> = vec![NONE; self.next_var_id];
        let mut exprs: Vec<LinearCombination> = Vec::new();
        let mut eliminated: Vec<usize> = Vec::new();
        // Which expressions read each wire, so that eliminating a wire an
        // earlier expression depends on can be pushed back into it immediately.
        //
        // The first version of this pass instead *pinned* those wires - refused
        // to eliminate them for the rest of the round - and left the rest to the
        // next iteration of the fixpoint. That is correct but quadratic in the
        // wrong place: a Poseidon permutation is a chain of linear layers, so
        // each round could only peel one level off it, and the pass hit the
        // 10-iteration safety cap having done a full sweep over every constraint
        // and every witness recipe ten times. Back-substitution finishes the
        // whole chain in one.
        // Allocated on the first elimination, not up front. Y's own front end
        // folds linear combinations as it builds them, so this pass finds
        // nothing there and would otherwise charge it 18 MB of zeroed `Vec`
        // headers per round for the privilege.
        let mut uses: Vec<Vec<u32>> = Vec::new();
        let mut drop_constraint: Vec<bool> = vec![false; self.constraints.len()];

for i in 0..self.constraints.len() {
            let c = &self.constraints[i];
            if c.c.terms.len() != 1 {
                continue;
            }
            let (w, coeff_w) = (c.c.terms[0].0, c.c.terms[0].1);
            if w == 0 || w >= sub_idx.len() || sub_idx[w] != NONE {
                continue;
            }

            // `A * B` must be linear, i.e. one side is a bare constant. Tested
            // before the two set lookups because it is a couple of array reads
            // against a hash, and on a circuit built from `a * b` products it
            // rejects almost everything.
            let (k, lin) = if let Some(k) = c.a.is_constant() {
                (k, &c.b)
            } else if let Some(k) = c.b.is_constant() {
                (k, &c.a)
            } else {
                continue;
            };
            if boundary.contains(&w) || opaque.contains(&w) {
                continue;
            }
            if k.is_zero() {
                // `0 = c_w * w` pins `w` to zero; that is a definition, but the
                // dedicated `Const` recipe path below expects a real expression
                // and this shape is rare enough not to special-case.
                continue;
            }
            // The pivot must not appear on the product side as well - then the
            // constraint does not define it in the shape the witness scan reads,
            // and the coefficient arithmetic below would be wrong.
            if lin.terms.iter().any(|(t, _)| *t == w) {
                continue;
            }

            // w = k * lin / coeff_w, with any wire eliminated earlier in this
            // round already resolved so the expression never dangles.
            //
            // `Fr::inv` is Fermat's little theorem - a 254-bit exponentiation,
            // ~380 Montgomery multiplies, ~6 us. Calling it once per candidate
            // constraint was *the* cost of this pass: 0.45 s of its 0.55 s on a
            // 200-hash chain, for a coefficient that a `<==` always leaves as
            // exactly 1. The identity checks below are not micro-optimisation,
            // they are the difference between the pass paying for itself and not.
            let one = Fr::one();
            let scale = if coeff_w == one { k } else { k.mul(&coeff_w.inv()) };
            let unit = scale == one;
            let mut expr = LinearCombination::zero();
            for (t, tc) in &lin.terms {
                let c2 = if unit { *tc } else { tc.mul(&scale) };
                match sub_idx.get(*t).copied() {
                    Some(idx) if idx != NONE => expr.add_linear(&exprs[idx as usize], c2),
                    _ => expr.add_term(*t, c2),
                }
            }
            expr.simplify();
            // Load-bearing, not defensive. `w2 * 1 = w1` followed by
            // `w1 * 1 = w2` states the same equality twice; having eliminated
            // `w1` as `w2`, resolving the second constraint yields `w2 = w2`.
            // Recording that would give `w2` a recipe that reads itself, which
            // the witness pass cannot evaluate and Kahn's sort would report as a
            // cycle. Keeping the second constraint instead costs one constraint
            // and is always correct.
            if expr.terms.iter().any(|(t, _)| *t == w) {
                continue;
            }

            // Fill-in guard. Replacing `w` by an n-term expression at each of
            // its remaining uses adds roughly `(uses-1)*(n-1)` terms. Unbounded,
            // this densifies things like a 32-bit recomposition into every
            // constraint that touches one of its bits - correct, and much slower
            // to prove. The overwhelmingly common case, `out <== in`, has n = 1
            // and costs nothing.
            if occ.is_empty() {
                occ = vec![0; self.next_var_id];
                for c in &self.constraints {
                    for lc in [&c.a, &c.b, &c.c] {
                        for (t, _) in &lc.terms {
                            if *t != 0 && *t < occ.len() {
                                occ[*t] += 1;
                            }
                        }
                    }
                }
            }
            let sites = occ[w] as usize + uses.get(w).map_or(0, Vec::len);
            let added = sites.saturating_sub(1) * expr.terms.len().saturating_sub(1);
            if added > budget {
                continue;
            }

            if uses.is_empty() {
                uses = vec![Vec::new(); self.next_var_id];
            }
            let idx = exprs.len() as u32;
            for (t, _) in &expr.terms {
                if *t != 0 {
                    uses[*t].push(idx);
                }
            }
            exprs.push(expr);
            sub_idx[w] = idx;
            eliminated.push(w);
            drop_constraint[i] = true;

            // Push the new definition back into the expressions that were
            // written in terms of `w`. Those are exactly `uses[w]`, so this is
            // targeted rather than a re-scan, and it keeps the invariant every
            // step below relies on: no stored expression ever mentions a wire
            // that has been eliminated.
            let dependents = std::mem::take(&mut uses[w]);
            for j in dependents {
                let (before, rest) = exprs.split_at_mut(idx as usize);
                let (target, source) = (&mut before[j as usize], &rest[0]);
                if !substitute_one(target, w, source) {
                    continue;
                }
                for (t, _) in &source.terms {
                    if *t != 0 {
                        uses[*t].push(j);
                    }
                }
            }
        }

        if eliminated.is_empty() {
            return false;
        }

let mut idx = 0;
        self.constraints.retain(|_| {
            let keep = !drop_constraint[idx];
            idx += 1;
            keep
        });
for c in &mut self.constraints {
            substitute_lc(&mut c.a, &sub_idx, &exprs);
            substitute_lc(&mut c.b, &sub_idx, &exprs);
            substitute_lc(&mut c.c, &sub_idx, &exprs);
        }
        // Same obligation as `remap_witness_recipes`: the recipes hold linear
        // combinations captured at emit time, and a recipe still reading an
        // eliminated wire evaluates it as zero.
        //
        // An eliminated wire's own recipe is skipped because it is overwritten
        // below - and on circom input that is most of them, since every `<==`
        // leaves a recipe behind and this pass exists to eliminate exactly those
        // wires.
for (wire, op) in self.witness_recipes.iter_mut() {
            if sub_idx.get(*wire).copied().unwrap_or(NONE) != NONE {
                continue;
            }
            substitute_witness_op(op, &sub_idx, &exprs);
        }

        // The eliminated wires stay in the variable table - dropping them would
        // renumber every wire - but they still need a value, because the `.wtns`
        // has a slot for each. Their recipes are the expression the deleted
        // constraint asserted, so the witness a caller gets is the same one the
        // unoptimised circuit would have produced, which is what lets
        // `zk_linear_substitution.rs` compare the two directly.
        //
        // Written after the substitution loop above, since they are already
        // expressed in surviving wires.
let one = LinearCombination::constant(Fr::one());
        for (n, w) in eliminated.iter().enumerate() {
            // Moved, not cloned: `exprs` has served its purpose by this point,
            // and cloning here is one allocation per eliminated wire - about
            // 94,000 of them on a 200-hash chain.
            let expr = std::mem::replace(&mut exprs[n], LinearCombination::zero());
            self.witness_recipes.insert(*w, WitnessOp::MulLc(expr, one.clone()));
        }

        // Substitution can turn a constraint into `0 * x = 0`. Those are
        // vacuous and go; a constraint that reduces to a NON-zero constant
        // identity is unsatisfiable and stays, because deleting it would turn an
        // impossible circuit into a provable one.
self.constraints.retain(|c| !constraint_is_vacuous(c));
        true
    }

    /// Renumber the wires densely, dropping the ones nothing refers to any more.
    ///
    /// # Why the circuit has dead wires at all
    ///
    /// The two reduction passes above both abandon wires by design.
    /// `substitute_linear_constraints` deletes the constraint that *defined* an
    /// intermediate and rewrites its uses into the defining expression;
    /// `dedup_identical_products` points two products at one wire and abandons
    /// the loser. Neither renumbers, and neither can: they run to a fixpoint,
    /// and renumbering mid-round would invalidate the `occ`/`uses`/`sub_idx`
    /// arrays the round is indexing by wire id. So the wire count stayed at
    /// whatever the front end allocated - 153,605 on a 200-hash Poseidon chain
    /// against circom's ~103,000, *after* Y had already emitted 1.86x fewer
    /// constraints than circom for the same circuit.
    ///
    /// That gap is not cosmetic. Groth16's proving key has a G1 element per
    /// wire and its MSMs run over one scalar per wire, so a dead wire costs a
    /// curve operation in every proof, forever. It is the whole difference
    /// between the 2.7x constraint reduction and the 1.4x prove speedup
    /// measured in `zk_linear_substitution::what_the_reduction_buys_at_proving_time`.
    ///
    /// # Why it is sound
    ///
    /// A wire that appears in no surviving constraint is existentially
    /// quantified and unconstrained: the system's satisfying assignments are a
    /// product of the surviving wires' solutions with *every* value of the dead
    /// one. Projecting it away preserves satisfiability in both directions, and
    /// it cannot change the statement, because the boundary - `1`, the public
    /// inputs, the private inputs and the outputs - is marked live outright and
    /// is exactly what a verifier sees.
    ///
    /// The one subtlety is that liveness is not just "mentioned in a
    /// constraint". A live wire's *witness recipe* may read a wire that no
    /// constraint mentions, so liveness is closed transitively over
    /// `witness_op_deps` (and `witness_op_writes`, so a `HintBlock` cannot end
    /// up with half its outputs deleted). Dropping a wire the solver still
    /// needed would not produce a wrong proof - it would produce no proof, the
    /// same shape of failure the CSE pass caused when it forgot the recipes.
    ///
    /// # The renumbering must be stable
    ///
    /// New ids ascend with old ones. `execute_host_witness_ir` assigns the
    /// public and private inputs POSITIONALLY in wire order, so a map that
    /// reordered them would feed the circuit its arguments shuffled - a wrong
    /// answer with nothing to catch it, since every wire would still be defined.
    ///
    /// Returns whether anything was dropped.
    fn compact_wires(&mut self) -> bool {
        if !compaction_enabled() {
            return false;
        }
        let n = self.next_var_id;
        if n == 0 {
            return false;
        }

        let mut live = vec![false; n];
        live[0] = true; // the constant-1 wire

        for &w in self
            .public_inputs
            .iter()
            .chain(self.private_inputs.iter())
            .chain(self.outputs.iter())
        {
            if w < n {
                live[w] = true;
            }
        }

        // Under-constrained hint wires are kept deliberately. They are what
        // error[Z0042] reports, and `run_optimizer()` lets a front end reach
        // this pass without having run that check yet - deleting the evidence
        // would turn a refused circuit into an accepted one.
        for &w in self.unconstrained_hint_vars.keys() {
            if w < n {
                live[w] = true;
            }
        }

        for c in &self.constraints {
            for lc in [&c.a, &c.b, &c.c] {
                for (t, _) in &lc.terms {
                    if *t < n {
                        live[*t] = true;
                    }
                }
            }
        }

        // Transitive closure through the recipes, as above.
        let mut stack: Vec<usize> = (0..n).filter(|&i| live[i]).collect();
        let mut touched = Vec::new();
        while let Some(w) = stack.pop() {
            let Some(op) = self.witness_recipes.get(&w) else {
                continue;
            };
            touched.clear();
            Self::witness_op_deps(op, &mut touched);
            Self::witness_op_writes(op, &mut touched);
            for &d in &touched {
                if d < n && !live[d] {
                    live[d] = true;
                    stack.push(d);
                }
            }
        }

        let live_count = live.iter().filter(|b| **b).count();
        if live_count == n {
            return false;
        }

        const DEAD: usize = usize::MAX;
        let mut map = vec![DEAD; n];
        let mut next = 0;
        for (old, &alive) in live.iter().enumerate() {
            if alive {
                map[old] = next;
                next += 1;
            }
        }

        // Every consumer of a wire id, rewritten together.
        //
        // This is the obligation the design-rule table in CLAUDE.md records
        // against `optimize_circuit`: a pass that renumbers wires has more than
        // one output, and updating only some of them is the same failure as
        // handling only some arms of a match. The full list of things indexed by
        // wire id is the constraints, the witness recipes (keys *and* the linear
        // combinations inside them), the variable-name table, the three boundary
        // lists, the hint-variable map, and `next_var_id`.
        for c in &mut self.constraints {
            replace_wires_in_lc(&mut c.a, &map);
            replace_wires_in_lc(&mut c.b, &map);
            replace_wires_in_lc(&mut c.c, &map);
        }

        let recipes = std::mem::take(&mut self.witness_recipes);
        self.witness_recipes = recipes
            .into_iter()
            .filter(|(w, _)| map.get(*w).copied().unwrap_or(DEAD) != DEAD)
            .map(|(w, mut op)| {
                remap_witness_op(&mut op, &map);
                (map[w], op)
            })
            .collect();

        let mut names = Vec::with_capacity(live_count);
        for (old, &alive) in live.iter().enumerate() {
            if alive {
                names.push(match self.variables.get_mut(old) {
                    Some(s) => std::mem::take(s),
                    None => format!("wire_{}", old),
                });
            }
        }
        self.variables = names;

        // The boundary is marked live unconditionally above, so none of these
        // can be dead. Stated rather than assumed, because the failure if it
        // ever were is a `usize::MAX` stored in `public_inputs` - which does not
        // fault here, but indexes off the end of the witness in whichever
        // consumer touches it first, arbitrarily far from this pass.
        for list in [
            &mut self.public_inputs,
            &mut self.private_inputs,
            &mut self.outputs,
        ] {
            for w in list.iter_mut() {
                let new = map[*w];
                assert!(
                    new != DEAD,
                    "compact_wires dropped boundary wire {} - the circuit's \
                     interface is part of the statement it proves",
                    *w
                );
                *w = new;
            }
        }

        let hints = std::mem::take(&mut self.unconstrained_hint_vars);
        self.unconstrained_hint_vars = hints
            .into_iter()
            .filter(|(w, _)| map.get(*w).copied().unwrap_or(DEAD) != DEAD)
            .map(|(w, v)| (map[w], v))
            .collect();

        self.next_var_id = live_count;

        // A dead id surviving into a constraint would be `usize::MAX`, which
        // indexes nothing and would panic far from here. The liveness closure
        // above makes it impossible; this is what says so out loud if a future
        // `WitnessOp` variant is added to one of the two walkers and not the
        // other.
        debug_assert!(
            self.constraints.iter().all(|c| [&c.a, &c.b, &c.c]
                .iter()
                .all(|lc| lc.terms.iter().all(|(t, _)| *t < live_count))),
            "compact_wires left a dangling wire id in a constraint"
        );

        true
    }


    // ---- builder API ----
    //
    // Exposed for front ends that are not Y's own AST walker - the circom front
    // end drives these directly. Everything after constraint construction (the
    // CSE pass, the snarkjs wire map, the `.r1cs`/`.wtns`/`.sym` writers, the
    // witness solver) is language-agnostic and is the part worth reusing.

    /// Allocate a fresh wire. `name` appears in the `.sym` file.
    pub fn alloc_wire(&mut self, name: &str) -> usize {
        self.new_wire(name)
    }

    /// Append a constraint `a * b = c`.
    pub fn push_constraint(&mut self, a: LinearCombination, b: LinearCombination, c: LinearCombination) {
        self.add_constraint(Constraint { a, b, c, span: None });
    }

    /// Record how the witness pass should compute `wire`.
    ///
    /// Required whenever the defining constraint has more than one unknown -
    /// which is every gadget, and every `a*b + c` fused into one constraint.
    pub fn set_witness_recipe(&mut self, wire: usize, op: WitnessOp) {
        self.witness_recipes.insert(wire, op);
    }

    /// Run the constraint-reduction pass. Front ends call this once, after
    /// emitting everything.
    pub fn run_optimizer(&mut self) {
        self.optimize_circuit();
    }

    /// The emitter's own circuit, borrowed. Identical content to
    /// `build_circuit()`, without duplicating the constraint list.
    pub fn view(&self) -> CircuitView<'_> {
        CircuitView {
            num_variables: self.next_var_id,
            variables: &self.variables,
            public_inputs: &self.public_inputs,
            private_inputs: &self.private_inputs,
            outputs: &self.outputs,
            constraints: &self.constraints,
        }
    }

    pub fn build_circuit(&self) -> Circuit {
        Circuit {
            num_variables: self.next_var_id,
            variables: self.variables.clone(),
            public_inputs: self.public_inputs.clone(),
            private_inputs: self.private_inputs.clone(),
            outputs: self.outputs.clone(),
            constraints: self.constraints.clone(),
        }
    }

    /// The circuit's private input names, in declaration order.
    ///
    /// `new_wire` appends `_<id>` to keep wire names unique, so the suffix is
    /// stripped back off here. The order is the order `solve_r1cs_witness`
    /// consumes its `priv_in` slice, which is what makes it safe to map a named
    /// input file onto that slice.
    pub fn private_input_names(&self) -> Vec<String> {
        self.private_inputs
            .iter()
            .map(|w| {
                let n = &self.variables[*w];
                n.rsplit_once('_').map(|(head, _)| head.to_string()).unwrap_or_else(|| n.clone())
            })
            .collect()
    }

    /// Writes a `.wtns` witness file in iden3's format, the one snarkjs reads.
    ///
    /// Layout mirrors `.r1cs`: magic, version, section count, then
    /// length-prefixed sections. Section 1 is the header (field size, prime,
    /// witness count) and section 2 is the values, each little-endian and
    /// padded to the field size.
    ///
    /// Without this, Y could emit a constraint system snarkjs accepts but never
    /// a proof it could produce - `snarkjs groth16 prove` needs both files.
    pub fn write_wtns_binary(
        circuit: &Circuit,
        witness: &[Fr],
        output_path: &str,
    ) -> std::io::Result<()> {
        Self::write_wtns_binary_view(circuit.view(), witness, output_path)
    }

    pub fn write_wtns_binary_view(
        circuit: CircuitView<'_>,
        witness: &[Fr],
        output_path: &str,
    ) -> std::io::Result<()> {
        use std::io::Write as IoWrite;
        let file = std::fs::File::create(output_path)?;
        let mut writer = std::io::BufWriter::new(file);

        writer.write_all(b"wtns")?;
        writer.write_all(&2u32.to_le_bytes())?; // version
        writer.write_all(&2u32.to_le_bytes())?; // nSections

        let n8: u32 = 32;
        let mut header = Vec::new();
        header.write_all(&n8.to_le_bytes())?;
        header.write_all(&Fr::modulus().to_bytes_le(n8 as usize))?;
        header.write_all(&(witness.len() as u32).to_le_bytes())?;

        writer.write_all(&1u32.to_le_bytes())?;
        writer.write_all(&(header.len() as u64).to_le_bytes())?;
        writer.write_all(&header)?;

        // Permuted into the same order the .r1cs constraints were written in.
        let (old_to_new, _, _, _) = Self::snarkjs_wire_map_view(circuit);
        let mut permuted = vec![Fr::zero(); witness.len()];
        for (old, new) in &old_to_new {
            if *old < witness.len() && *new < permuted.len() {
                permuted[*new] = witness[*old].clone();
            }
        }

        let mut data = Vec::with_capacity(permuted.len() * n8 as usize);
        for w in &permuted {
            data.extend_from_slice(&w.to_bytes_le(n8 as usize));
        }
        writer.write_all(&2u32.to_le_bytes())?;
        writer.write_all(&(data.len() as u64).to_le_bytes())?;
        writer.write_all(&data)?;

        writer.flush()
    }

    /// Y's wire numbering permuted into iden3's convention.
    ///
    /// snarkjs requires a specific layout - `1`, then outputs, public inputs,
    /// private inputs, and finally intermediates - whereas Y allocates wires in
    /// the order the emitter happens to need them (inputs first, the output
    /// typically last). The `.r1cs` writer has always applied this permutation
    /// to its constraint terms.
    ///
    /// It is shared with `write_wtns_binary` precisely because it must be. When
    /// the witness was written in Y's raw order against constraints written in
    /// iden3's, both files were internally consistent and `snarkjs wtns check`
    /// rejected the pair at constraint 1. Y's own solver saw nothing wrong,
    /// because it works entirely in Y's numbering. Two orderings, one of them
    /// only ever exercised by an external tool.
    ///
    /// Returns the map plus the three counts the `.r1cs` header declares.
    pub fn snarkjs_wire_map(circuit: &Circuit) -> (HashMap<usize, usize>, usize, usize, usize) {
        Self::snarkjs_wire_map_view(circuit.view())
    }

    pub fn snarkjs_wire_map_view(circuit: CircuitView<'_>) -> (HashMap<usize, usize>, usize, usize, usize) {
        let mut old_to_new = HashMap::new();
        old_to_new.insert(0, 0); // the constant-1 wire
        let mut next_new_id = 1;

        for &w in circuit.outputs {
            old_to_new.entry(w).or_insert_with(|| {
                let id = next_new_id;
                next_new_id += 1;
                id
            });
        }
        let n_pub_out = next_new_id - 1;

        for &w in circuit.public_inputs {
            old_to_new.entry(w).or_insert_with(|| {
                let id = next_new_id;
                next_new_id += 1;
                id
            });
        }
        let n_pub_in = next_new_id - 1 - n_pub_out;

        for &w in circuit.private_inputs {
            old_to_new.entry(w).or_insert_with(|| {
                let id = next_new_id;
                next_new_id += 1;
                id
            });
        }
        let n_prv_in = next_new_id - 1 - n_pub_out - n_pub_in;

        let mut aux_wires: Vec<usize> = (1..circuit.num_variables)
            .filter(|w| !old_to_new.contains_key(w))
            .collect();
        aux_wires.sort();
        for w in aux_wires {
            old_to_new.insert(w, next_new_id);
            next_new_id += 1;
        }

        (old_to_new, n_pub_out, n_pub_in, n_prv_in)
    }

    pub fn write_r1cs_binary(&self, output_path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::BufWriter;
        use std::io::Write;

        let circuit = self.view();

        // 1. Determine the old-to-new wire mapping
        let (old_to_new, n_pub_out, n_pub_in, n_prv_in) = Self::snarkjs_wire_map_view(circuit);

        // 2. Write binary .r1cs file
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);
        let encoder = R1csEncoder::new();
        encoder.encode_to_stream(circuit, &old_to_new, n_pub_out, n_pub_in, n_prv_in, &mut writer)?;

        // 3. Write symbols (.sym) file
        let sym_path = format!("{}.sym", output_path.strip_suffix(".r1cs").unwrap_or(output_path));
        let sym_file = File::create(&sym_path)?;
        let mut sym_writer = BufWriter::new(sym_file);

        let mut entries = Vec::new();
        for (&old, &new) in &old_to_new {
            if old < circuit.variables.len() {
                entries.push((old, new, &circuit.variables[old]));
            }
        }
        entries.sort_by_key(|e| e.1);
        for (old, new, name) in entries {
            writeln!(sym_writer, "{},{},0,{}", old, new, name)?;
        }

        Ok(())
    }
}

pub struct R1csEncoder;

impl R1csEncoder {
    pub fn new() -> Self {
        Self
    }

    pub fn encode_to_stream<W: std::io::Write>(
        &self,
        circuit: CircuitView<'_>,
        old_to_new: &HashMap<usize, usize>,
        n_pub_out: usize,
        n_pub_in: usize,
        n_prv_in: usize,
        writer: &mut W,
    ) -> std::io::Result<()> {
        use std::io::Write as IoWrite;

        // Write Magic: "r1cs"
        writer.write_all(b"r1cs")?;

        // Write Version: 1
        writer.write_all(&1u32.to_le_bytes())?;

        // Write nSections: 3 (Header, Constraints, Wire2Label)
        writer.write_all(&3u32.to_le_bytes())?;

        // --- 1. HEADER SECTION ---
        let mut header_buf = Vec::new();
        let fs = 32u32;
        header_buf.write_all(&fs.to_le_bytes())?;

        let prime_bytes = Fr::modulus().to_bytes_le(32);
        header_buf.write_all(&prime_bytes)?;

        let n_wires = circuit.num_variables;
        header_buf.write_all(&(n_wires as u32).to_le_bytes())?;
        header_buf.write_all(&(n_pub_out as u32).to_le_bytes())?;
        header_buf.write_all(&(n_pub_in as u32).to_le_bytes())?;
        header_buf.write_all(&(n_prv_in as u32).to_le_bytes())?;

        let n_labels = n_wires as u64;
        header_buf.write_all(&n_labels.to_le_bytes())?;
        header_buf.write_all(&(circuit.constraints.len() as u32).to_le_bytes())?;

        self.write_section(writer, 1, &header_buf)?;

        // --- 2. CONSTRAINTS SECTION ---
        //
        // Streamed, not buffered. This used to build the entire section in a
        // `Vec<u8>` before writing a byte of it - on a dense circuit that is
        // ~36 bytes per term held in memory on top of the constraints
        // themselves, and it grows by doubling, so the transient peak is worse
        // again. The section length the format wants up front is arithmetic, not
        // something that has to be discovered by serialising first.
        let section_len: u64 = circuit
            .constraints
            .iter()
            .map(|c| {
                let terms = c.a.terms.len() + c.b.terms.len() + c.c.terms.len();
                // 3 x u32 term-count prefix, then (u32 wire + 32-byte coeff)
                3 * 4 + terms * (4 + 32)
            })
            .sum::<usize>() as u64;

        writer.write_all(&2u32.to_le_bytes())?;
        writer.write_all(&section_len.to_le_bytes())?;

        // Reused across every linear combination, so the remapping scratch
        // allocates once for the whole file rather than once per LC, and the
        // coefficient never gets a `Vec` of its own at all.
        let mut remapped: Vec<(u32, Fr)> = Vec::new();
        let mut coeff_bytes = [0u8; 32];
        let mut written: u64 = 0;

        for c in circuit.constraints {
            for lc in [&c.a, &c.b, &c.c] {
                remapped.clear();
                remapped.extend(lc.terms.iter().map(|&(old_wire, coeff)| {
                    (*old_to_new.get(&old_wire).unwrap_or(&0) as u32, coeff)
                }));
                // Sort by new wire ID ascending
                remapped.sort_by_key(|t| t.0);

                writer.write_all(&(remapped.len() as u32).to_le_bytes())?;
                written += 4;
                for (wire_id, coeff) in &remapped {
                    coeff.write_bytes_le(&mut coeff_bytes);
                    writer.write_all(&wire_id.to_le_bytes())?;
                    writer.write_all(&coeff_bytes)?;
                    written += 36;
                }
            }
        }

        // The length was declared before the body was written, so a mismatch
        // would produce a structurally corrupt `.r1cs` that snarkjs reads as
        // garbage from the next section onward rather than rejecting outright.
        debug_assert_eq!(written, section_len, "declared .r1cs section length does not match what was written");
        if written != section_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "internal error: .r1cs constraints section declared {} bytes but wrote {}",
                    section_len, written
                ),
            ));
        }

        // --- 3. WIRE TO LABEL MAP SECTION ---
        let mut map_buf = Vec::new();
        let mut new_to_old = vec![0u64; n_wires];
        for (&old, &new) in old_to_new {
            if new < n_wires {
                new_to_old[new] = old as u64;
            }
        }
        for label_id in new_to_old {
            map_buf.write_all(&label_id.to_le_bytes())?;
        }
        self.write_section(writer, 3, &map_buf)?;

        Ok(())
    }

    fn write_section<W: std::io::Write>(
        &self,
        writer: &mut W,
        section_type: u32,
        content: &[u8],
    ) -> std::io::Result<()> {
        writer.write_all(&section_type.to_le_bytes())?;
        let size = content.len() as u64;
        writer.write_all(&size.to_le_bytes())?;
        writer.write_all(content)?;
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biguint_ops() {
        let p_str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";
        let p = BigUint::from_str(p_str);
        
        let zero = BigUint::zero();
        let one = BigUint::one();
        
        assert!(zero < one);
        assert!(one < p);
        
        // Addition
        let p_plus_one = p.add(&one);
        assert_eq!(p_plus_one.sub(&one), p);

        // Multiplication
        let two = BigUint::from_u64(2);
        let doubled = p.mul(&two);
        assert_eq!(doubled.sub(&p), p);
    }

    #[test]
    fn test_field_ops() {
        let zero = Fr::from_u64(0);
        let one = Fr::from_u64(1);
        let neg_one = zero.sub(&one);
        
        // neg_one + 1 = 0
        assert_eq!(neg_one.add(&one), zero);
    }

    #[test]
    fn test_configurable_fields() {
        for f in &[ScalarField::Bn254, ScalarField::Bls12_381, ScalarField::Pallas, ScalarField::Vesta] {
            let config = FieldConfig::get(*f);
            set_active_modulus(&config.p);
            
            let zero = Fr::from_u64(0);
            let one = Fr::from_u64(1);
            let neg_one = zero.sub(&one);
            
            // neg_one + 1 = 0
            assert_eq!(neg_one.add(&one), zero);
            
            // modular inverse: 2 * inv(2) = 1
            let two = Fr::from_u64(2);
            let inv_two = two.inv();
            assert_eq!(two.mul(&inv_two), one);
        }
    }

    #[test]
    fn test_end_to_end_zk_target() {
        // This exercises `@zk_target` field/scheme selection. It used to call
        // `poseidon_hash` here, which is now refused over Pallas - Y only has
        // circomlib's BN254 parameters, and there is no such thing as a
        // field-agnostic Poseidon. See `poseidon_over_non_bn254_is_refused`.
        let source = r#"
        @zk_target(field = "pallas", scheme = "r1cs", opt_level = 1)
        module PallasCircuit {
            fn main(x: I32, y: I32) -> I32 {
                @bounds(min=0, max=15)
                let bounded_var = x;
                let scaled = bounded_var * y;
                return scaled + y;
            }
        }
        "#;
        
        let mut sub_lexer = crate::lexer::Lexer::new(source);
        let sub_tokens = sub_lexer.tokenize();
        let mut sub_parser = crate::parser::Parser::new(sub_tokens);
        let prog = sub_parser.parse_program().unwrap();

        let mut emitter = ZkEmitter::new();
        let r1cs_text = emitter.emit_program(&prog).unwrap();

        assert!(r1cs_text.contains("Curve Field: PALLAS"));
        assert!(r1cs_text.contains("Modulus r: 28948022309329048855892746252171976963363056481941560715954676764349967630337"));
        assert!(r1cs_text.contains("Proof Scheme: R1cs"));
        assert!(emitter.constraints.len() > 0);
    }

    /// Poseidon must refuse a field it has no parameters for.
    ///
    /// The round constants and MDS are derived per-prime; running BN254's over
    /// Pallas does not give a different-but-valid hash, it gives one nobody has
    /// analysed and nobody else can reproduce. The previous implementation
    /// accepted every field and every arity by generating filler constants,
    /// which is the failure mode this pins shut.
    #[test]
    fn poseidon_over_non_bn254_is_refused() {
        let source = r#"
        @zk_target(field = "pallas", scheme = "r1cs", opt_level = 1)
        module PallasCircuit {
            fn main(x: I32, y: I32) -> I32 {
                return poseidon_hash(x, y);
            }
        }
        "#;
        let tokens = crate::lexer::Lexer::new(source).tokenize();
        let prog = crate::parser::Parser::new(tokens).parse_program().unwrap();
        let err = ZkEmitter::new()
            .emit_program(&prog)
            .expect_err("poseidon over Pallas must be refused");
        assert!(err.contains("BN254"), "error should name the supported field: {}", err);
    }

    #[test]
    fn test_bounded_while_loop() {
        let source = r#"
        @unsafe
        fn main(x: I32) -> I32 {
            let mut val = x;
            let mut count = 0;
            @max_iterations(5)
            while val < 10 {
                val = val + 2;
                count = count + 1;
            }
            return count;
        }
        "#;
        let mut sub_lexer = crate::lexer::Lexer::new(source);
        let sub_tokens = sub_lexer.tokenize();
        let mut sub_parser = crate::parser::Parser::new(sub_tokens);
        let prog = sub_parser.parse_program().unwrap();

        let mut emitter = ZkEmitter::new();
        let r1cs_text = emitter.emit_program(&prog).unwrap();
        assert!(r1cs_text.contains("active_mask"));
        assert!(emitter.constraints.len() > 0);
    }

    #[test]
    fn test_unbounded_while_error() {
        let source = r#"
        @unsafe
        fn main(x: I32) -> I32 {
            let mut val = x;
            while val > 0 {
                val = val - 1;
            }
            return val;
        }
        "#;
        let mut sub_lexer = crate::lexer::Lexer::new(source);
        let sub_tokens = sub_lexer.tokenize();
        let mut sub_parser = crate::parser::Parser::new(sub_tokens);
        let prog = sub_parser.parse_program().unwrap();

        let mut emitter = ZkEmitter::new();
        let err = emitter.emit_program(&prog).unwrap_err();
        assert!(err.contains("Z0010"));
    }

    #[test]
    fn test_static_const_recursion() {
        let source = r#"
        @unsafe
        fn sum_const(n: I32, x: I32) -> I32 {
            if n == 0 {
                return 0;
            } else {
                return x + sum_const(n - 1, x);
            }
        }

        @unsafe
        fn main(x: I32) -> I32 {
            return sum_const(4, x);
        }
        "#;
        let mut sub_lexer = crate::lexer::Lexer::new(source);
        let sub_tokens = sub_lexer.tokenize();
        let mut sub_parser = crate::parser::Parser::new(sub_tokens);
        let prog = sub_parser.parse_program().unwrap();

        let mut emitter = ZkEmitter::new();
        let r1cs_text = emitter.emit_program(&prog).unwrap();
        assert!(emitter.constraints.len() > 0);
        assert!(r1cs_text.contains("Proof Scheme: R1cs"));
    }

    #[test]
    fn test_dynamic_recursion_error() {
        let source = r#"
        @unsafe
        fn rec_dyn(n: I32) -> I32 {
            if n == 0 {
                return 0;
            } else {
                return rec_dyn(n - 1);
            }
        }

        @unsafe
        fn main(x: I32) -> I32 {
            return rec_dyn(x);
        }
        "#;
        let mut sub_lexer = crate::lexer::Lexer::new(source);
        let sub_tokens = sub_lexer.tokenize();
        let mut sub_parser = crate::parser::Parser::new(sub_tokens);
        let prog = sub_parser.parse_program().unwrap();

        let mut emitter = ZkEmitter::new();
        let err = emitter.emit_program(&prog).unwrap_err();
        assert!(err.contains("Z0012"));
    }

    #[test]
    fn test_biguint_sub_underflow_modular_reduction() {
        let a = BigUint::from_u64(5);
        let b = BigUint::from_u64(10);
        let res = a.sub(&b);
        let p = active_modulus();
        let expected = p.sub(&BigUint::from_u64(5));
        assert_eq!(res, expected);
    }

    #[test]
    fn test_add_linear_term_deduplication() {
        let mut lc1 = LinearCombination::zero();
        lc1.add_term(1, Fr::from_u64(3));
        lc1.add_term(2, Fr::from_u64(2));
        lc1.simplify();

        let mut lc2 = LinearCombination::zero();
        lc2.add_term(1, Fr::from_u64(5));
        lc2.add_term(3, Fr::from_u64(1));
        lc2.simplify();

        lc1.add_linear(&lc2, Fr::one());
        assert!(!lc1.is_simplified);
        lc1.simplify();
        assert_eq!(lc1.terms.len(), 3);
        assert_eq!(lc1.terms[0], (1, Fr::from_u64(8)));
        assert_eq!(lc1.terms[1], (2, Fr::from_u64(2)));
        assert_eq!(lc1.terms[2], (3, Fr::from_u64(1)));
    }
}
