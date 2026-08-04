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
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

// ────────────────────────────────────────────────────────
// 1. BigUint and BN254 Fr Field Arithmetic
// ────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct BigUint {
    // Digits in little-endian order, base 2^32
    pub digits: Vec<u32>,
}

impl PartialOrd for BigUint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigUint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let lhs_len = self.effective_len();
        let rhs_len = other.effective_len();
        if lhs_len != rhs_len {
            return lhs_len.cmp(&rhs_len);
        }
        for i in (0..lhs_len).rev() {
            let d1 = self.digits.get(i).copied().unwrap_or(0);
            let d2 = other.digits.get(i).copied().unwrap_or(0);
            if d1 != d2 {
                return d1.cmp(&d2);
            }
        }
        std::cmp::Ordering::Equal
    }
}

impl BigUint {
    pub fn zero() -> Self {
        Self { digits: vec![0] }
    }

    pub fn one() -> Self {
        Self { digits: vec![1] }
    }

    pub fn is_zero(&self) -> bool {
        self.digits.iter().all(|&d| d == 0)
    }

    pub fn from_u64(mut val: u64) -> Self {
        let mut digits = Vec::new();
        if val == 0 {
            digits.push(0);
        } else {
            while val > 0 {
                digits.push((val & 0xffffffff) as u32);
                val >>= 32;
            }
        }
        Self { digits }
    }

    pub fn trim(&mut self) {
        while self.digits.len() > 1 && *self.digits.last().unwrap() == 0 {
            self.digits.pop();
        }
    }

    pub fn effective_len(&self) -> usize {
        let mut len = self.digits.len();
        while len > 1 && self.digits[len - 1] == 0 {
            len -= 1;
        }
        len
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut digits = Vec::new();
        let mut carry = 0u64;
        let len = std::cmp::max(self.digits.len(), other.digits.len());
        for i in 0..len {
            let d1 = self.digits.get(i).cloned().unwrap_or(0) as u64;
            let d2 = other.digits.get(i).cloned().unwrap_or(0) as u64;
            let sum = d1 + d2 + carry;
            digits.push((sum & 0xffffffff) as u32);
            carry = sum >> 32;
        }
        if carry > 0 {
            digits.push(carry as u32);
        }
        let mut res = Self { digits };
        res.trim();
        res
    }

    pub fn sub(&self, other: &Self) -> Self {
        if self < other {
            let mod_val = ACTIVE_MODULUS.with(|m| m.borrow().clone());
            if !mod_val.is_zero() {
                let mut padded = self.clone();
                while padded < *other {
                    padded = padded.add(&mod_val);
                }
                let diff = padded.sub(other);
                let (_, rem) = diff.div_mod(&mod_val);
                return rem;
            }
        }
        let mut digits = Vec::new();
        let mut borrow = 0i64;
        let len = std::cmp::max(self.digits.len(), other.digits.len());
        for i in 0..len {
            let d1 = self.digits.get(i).cloned().unwrap_or(0) as i64;
            let d2 = other.digits.get(i).cloned().unwrap_or(0) as i64;
            let diff = d1 - d2 - borrow;
            if diff < 0 {
                digits.push((diff + 0x100000000) as u32);
                borrow = 1;
            } else {
                digits.push(diff as u32);
                borrow = 0;
            }
        }
        if borrow > 0 {
            panic!("BigUint subtraction underflow");
        }
        let mut res = Self { digits };
        res.trim();
        res
    }

    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut digits = vec![0u32; self.digits.len() + other.digits.len()];
        for i in 0..self.digits.len() {
            let mut carry = 0u64;
            for j in 0..other.digits.len() {
                let prod = (self.digits[i] as u64) * (other.digits[j] as u64) + (digits[i + j] as u64) + carry;
                digits[i + j] = (prod & 0xffffffff) as u32;
                carry = prod >> 32;
            }
            if carry > 0 {
                digits[i + other.digits.len()] += carry as u32;
            }
        }
        let mut res = Self { digits };
        res.trim();
        res
    }

    /// `self / (2^32)^n` - drop the `n` least significant limbs.
    pub fn shr_limbs(&self, n: usize) -> Self {
        if n >= self.digits.len() {
            return Self::zero();
        }
        let mut res = Self { digits: self.digits[n..].to_vec() };
        res.trim();
        res
    }

    /// `self mod (2^32)^n` - keep the `n` least significant limbs.
    pub fn low_limbs(&self, n: usize) -> Self {
        let take = n.min(self.digits.len());
        let mut res = Self { digits: self.digits[..take].to_vec() };
        if res.digits.is_empty() {
            res.digits.push(0);
        }
        res.trim();
        res
    }

    pub fn bit_len(&self) -> usize {
        if self.is_zero() {
            return 0;
        }
        let last_idx = self.digits.len() - 1;
        let last_digit = self.digits[last_idx];
        let bits_in_last = 32 - last_digit.leading_zeros() as usize;
        last_idx * 32 + bits_in_last
    }

    pub fn get_bit(&self, bit_idx: usize) -> bool {
        let digit_idx = bit_idx / 32;
        let shift = bit_idx % 32;
        if digit_idx >= self.digits.len() {
            false
        } else {
            ((self.digits[digit_idx] >> shift) & 1) == 1
        }
    }

    pub fn set_bit(&mut self, bit_idx: usize, val: bool) {
        let digit_idx = bit_idx / 32;
        let shift = bit_idx % 32;
        while self.digits.len() <= digit_idx {
            self.digits.push(0);
        }
        if val {
            self.digits[digit_idx] |= 1 << shift;
        } else {
            self.digits[digit_idx] &= !(1 << shift);
        }
        self.trim();
    }

    pub fn shl1(&self) -> Self {
        let mut digits = Vec::new();
        let mut carry = 0u32;
        for &d in &self.digits {
            digits.push((d << 1) | carry);
            carry = d >> 31;
        }
        if carry > 0 {
            digits.push(carry);
        }
        let mut res = Self { digits };
        res.trim();
        res
    }

    pub fn div_mod(&self, other: &Self) -> (Self, Self) {
        if other.is_zero() {
            panic!("Division by zero");
        }
        let mut quotient = Self::zero();
        let mut remainder = Self::zero();
        for i in (0..self.bit_len()).rev() {
            remainder = remainder.shl1();
            if self.get_bit(i) {
                remainder.set_bit(0, true);
            }
            if remainder >= *other {
                remainder = remainder.sub(other);
                quotient.set_bit(i, true);
            }
        }
        (quotient, remainder)
    }

    pub fn from_str(s: &str) -> Self {
        let mut res = Self::zero();
        let ten = Self::from_u64(10);
        for c in s.chars() {
            if let Some(digit) = c.to_digit(10) {
                res = res.mul(&ten).add(&Self::from_u64(digit as u64));
            }
        }
        res
    }

    /// Parse a hex literal, with or without a `0x` prefix.
    ///
    /// Separate from `from_str` on purpose. `from_str` reads DECIMAL and
    /// silently skips any character that is not a decimal digit, so feeding it
    /// `"0xee9a"` returns 0*10+... over just the `0` and `9` - a wrong number,
    /// no error. The Poseidon constants are hex, and a silently mis-parsed
    /// round constant is a hash that matches nothing.
    pub fn from_hex_str(s: &str) -> Result<Self, String> {
        let body = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
        if body.is_empty() {
            return Err(format!("empty hex literal: {:?}", s));
        }
        let mut res = Self::zero();
        let sixteen = Self::from_u64(16);
        for c in body.chars() {
            let digit = c
                .to_digit(16)
                .ok_or_else(|| format!("invalid hex digit {:?} in {:?}", c, s))?;
            res = res.mul(&sixteen).add(&Self::from_u64(digit as u64));
        }
        Ok(res)
    }

    /// Divide by a single 32-bit digit, returning quotient and remainder.
    ///
    /// One pass of 64-bit divides, versus `div_mod`'s bit-at-a-time loop.
    pub fn div_rem_small(&self, d: u32) -> (Self, u32) {
        debug_assert!(d != 0, "division by zero");
        let mut q = vec![0u32; self.digits.len()];
        let mut rem: u64 = 0;
        for i in (0..self.digits.len()).rev() {
            let cur = (rem << 32) | self.digits[i] as u64;
            q[i] = (cur / d as u64) as u32;
            rem = cur % d as u64;
        }
        let mut res = Self { digits: q };
        res.trim();
        (res, rem as u32)
    }

    /// Decimal rendering, 9 digits at a time.
    ///
    /// This used to peel one digit per iteration via `div_mod(10)`, and
    /// `div_mod` is a bit-by-bit long division - so each digit cost ~254
    /// allocating loop iterations, and a full 254-bit value cost ~20,000. That
    /// made *printing* the R1CS the single most expensive phase of ZK
    /// compilation: emitting one 241-constraint Poseidon spent 0.007s building
    /// the circuit and 1.3s formatting it. It stayed hidden because the
    /// polynomial benchmark circuit's coefficients are 0, 1 and 2, which are
    /// one iteration each; only dense constants like Poseidon's round keys
    /// expose it.
    pub fn to_decimal_string(&self) -> String {
        if self.is_zero() {
            return "0".to_string();
        }
        // 10^9 is the largest power of ten fitting in a u32.
        let mut temp = self.clone();
        let mut chunks = Vec::new();
        while !temp.is_zero() {
            let (q, r) = temp.div_rem_small(1_000_000_000);
            temp = q;
            chunks.push(r);
        }
        let mut s = chunks.last().unwrap().to_string();
        for chunk in chunks.iter().rev().skip(1) {
            s.push_str(&format!("{:09}", chunk));
        }
        s
    }

    pub fn to_bytes_le(&self, byte_len: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(byte_len);
        for &digit in &self.digits {
            bytes.extend_from_slice(&digit.to_le_bytes());
        }
        bytes.resize(byte_len, 0);
        bytes
    }
}


use std::cell::RefCell;

thread_local! {
    pub static ACTIVE_MODULUS: RefCell<BigUint> = RefCell::new(BigUint::from_str("21888242871839275222246405745257275088548364400416034343698204186575808495617"));
}

thread_local! {
    /// `(modulus, mu, k)` for the Barrett reduction below, recomputed only when
    /// the active field changes.
    static BARRETT: RefCell<Option<(BigUint, BigUint, usize)>> = const { RefCell::new(None) };
}

/// `x mod p` for any `x < b^(2k)`, where `b = 2^32` and `k` is `p`'s limb count.
///
/// Replaces a bit-by-bit long division that was costing the ZK backend
/// everything. `BigUint::div_mod` walks one bit at a time, and each step does a
/// shift, a compare and possibly a subtract - all allocating - so reducing a
/// 512-bit product ran 512 such rounds. A single `Fr::mul` measured **26 us**.
/// For scale, a competent BN254 multiply is tens of nanoseconds, and the
/// polynomial benchmark circuit only looked fast because its linear
/// combinations are one or two terms wide and it therefore barely multiplies.
/// Anything with dense constant coefficients - Poseidon above all - hit this
/// directly: one 241-constraint hash took 1.4s to emit, 13x slower than circom
/// compiling the same circuit.
///
/// Barrett needs no division at all: with `mu = floor(b^(2k) / p)` precomputed
/// once, an estimate of the quotient is two multiplications away, and the
/// estimate is never more than two off.
///
/// The correction loop is `while`, not `if`, deliberately. The standard bound
/// says at most two subtractions are needed for `x < b^(2k)`; looping costs
/// nothing when the bound holds and stays correct if it ever does not, which is
/// the right trade for code the soundness of every proof rests on.
pub fn barrett_reduce(x: &BigUint, p: &BigUint) -> BigUint {
    if x < p {
        let mut r = x.clone();
        r.trim();
        return r;
    }

    let (mu, k) = BARRETT.with(|cell| {
        let mut cell = cell.borrow_mut();
        if let Some((m, mu, k)) = cell.as_ref() {
            if m == p {
                return (mu.clone(), *k);
            }
        }
        // mu = floor(b^(2k) / p). Uses the slow path exactly once per field.
        let k = p.effective_len();
        let mut num = BigUint { digits: vec![0u32; 2 * k + 1] };
        num.digits[2 * k] = 1;
        let (mu, _) = num.div_mod(p);
        *cell = Some((p.clone(), mu.clone(), k));
        (mu, k)
    });

    // q3 = floor(floor(x / b^(k-1)) * mu / b^(k+1))
    let q1 = x.shr_limbs(k - 1);
    let q3 = q1.mul(&mu).shr_limbs(k + 1);

    // r = (x - q3*p) mod b^(k+1); the modulus keeps the subtraction in range.
    let r1 = x.low_limbs(k + 1);
    let r2 = q3.mul(p).low_limbs(k + 1);

    let mut r = if r1 >= r2 {
        r1.sub(&r2)
    } else {
        // Borrow b^(k+1) rather than going signed.
        let mut base = BigUint { digits: vec![0u32; k + 2] };
        base.digits[k + 1] = 1;
        r1.add(&base).sub(&r2)
    };

    while r >= *p {
        r = r.sub(p);
    }
    r.trim();
    r
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Fr(pub BigUint);

impl Fr {
    pub fn modulus() -> BigUint {
        ACTIVE_MODULUS.with(|m| m.borrow().clone())
    }

    #[inline(always)]
    pub fn with_modulus<R, F: FnOnce(&BigUint) -> R>(f: F) -> R {
        ACTIVE_MODULUS.with(|m| f(&m.borrow()))
    }

    pub fn zero() -> Self {
        Fr(BigUint::zero())
    }

    pub fn one() -> Self {
        Fr(BigUint::one())
    }

    pub fn from_u64(val: u64) -> Self {
        Self::with_modulus(|m| {
            let bi = BigUint::from_u64(val);
            let (_, r) = bi.div_mod(m);
            Fr(r)
        })
    }

    pub fn from_biguint(bi: BigUint) -> Self {
        Self::with_modulus(|m| {
            let (_, r) = bi.div_mod(m);
            Fr(r)
        })
    }

    pub fn add(&self, other: &Self) -> Self {
        Self::with_modulus(|modulus| {
            let sum = self.0.add(&other.0);
            if &sum >= modulus {
                Fr(sum.sub(modulus))
            } else {
                Fr(sum)
            }
        })
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self::with_modulus(|modulus| {
            if self.0 >= other.0 {
                Fr(self.0.sub(&other.0))
            } else {
                Fr(self.0.add(modulus).sub(&other.0))
            }
        })
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self::with_modulus(|modulus| Fr(barrett_reduce(&self.0.mul(&other.0), modulus)))
    }

    pub fn try_inv(&self) -> Result<Self, String> {
        if self.0.is_zero() {
            return Err("error[Z0040]: division by zero: finite field element zero has no modular inverse during host witness generation in @hint block".to_string());
        }
        Ok(self.inv())
    }

    pub fn inv(&self) -> Self {
        if self.0.is_zero() {
            panic!("Zero has no modular inverse");
        }
        Self::with_modulus(|p| {
            let mut t = BigUint::zero();
            let mut newt = BigUint::one();
            let mut r = p.clone();
            let mut newr = self.0.clone();

            let mut t_neg = false;
            let mut newt_neg = false;

            while !newr.is_zero() {
                let (quotient, remainder) = r.div_mod(&newr);
                r = newr;
                newr = remainder;

                let prod = quotient.mul(&newt);
                let next_t;
                let next_t_neg;

                if t_neg == newt_neg {
                    if t >= prod {
                        next_t = t.sub(&prod);
                        next_t_neg = t_neg;
                    } else {
                        next_t = prod.sub(&t);
                        next_t_neg = !t_neg;
                    }
                } else {
                    next_t = t.add(&prod);
                    next_t_neg = t_neg;
                }

                t = newt;
                t_neg = newt_neg;
                newt = next_t;
                newt_neg = next_t_neg;
            }

            if r > BigUint::one() {
                panic!("Modular inverse does not exist (GCD != 1 — modulus may not be prime)");
            }

            if t_neg {
                Fr(p.sub(&t))
            } else {
                Fr(t)
            }
        })
    }

    pub fn to_string(&self) -> String {
        self.0.to_decimal_string()
    }
}

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
            ScalarField::Vesta => (
                "vesta",
                "28948022309329048855892746252171976963363056481941600134020817490249052636161",
                254,
                31,
            ),
        };
        let p = BigUint::from_str(p_str);

        // Temporarily set active modulus so elements are reduced properly
        let prev_modulus = ACTIVE_MODULUS.with(|m| {
            let prev = m.borrow().clone();
            *m.borrow_mut() = p.clone();
            prev
        });

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
        ACTIVE_MODULUS.with(|m| {
            *m.borrow_mut() = prev_modulus;
        });

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
    Const(BigUint),
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
        if val.0.is_zero() {
            Self::zero()
        } else {
            Self { terms: vec![(0, val)], is_simplified: true }
        }
    }

    pub fn variable(id: usize) -> Self {
        Self { terms: vec![(id, Fr::one())], is_simplified: true }
    }

    pub fn add_constant(&mut self, val: Fr) {
        if !val.0.is_zero() {
            self.terms.push((0, val));
            self.is_simplified = false;
        }
    }

    pub fn add_term(&mut self, wire_id: usize, val: Fr) {
        if !val.0.is_zero() {
            self.terms.push((wire_id, val));
            self.is_simplified = false;
        }
    }

    pub fn add_linear(&mut self, other: &Self, scale: Fr) {
        if scale.0.is_zero() || other.terms.is_empty() {
            return;
        }
        let can_keep_simplified = self.terms.is_empty() && other.is_simplified;
        for (wire, coeff) in &other.terms {
            self.terms.push((*wire, coeff.mul(&scale)));
        }
        self.is_simplified = can_keep_simplified;
    }

    pub fn scale(&self, factor: Fr) -> Self {
        if factor.0.is_zero() {
            return Self::zero();
        }
        let terms = self.terms.iter().map(|(w, c)| (*w, c.mul(&factor))).collect();
        Self { terms, is_simplified: self.is_simplified }
    }

    pub fn simplify(&mut self) {
        if self.is_simplified {
            return;
        }
        if self.terms.len() <= 1 {
            if self.terms.len() == 1 && self.terms[0].1.0.is_zero() {
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
            let has_zeros = self.terms.iter().any(|(_, coeff)| coeff.0.is_zero());
            if !has_zeros {
                self.is_simplified = true;
                return; // Already sorted, distinct, and non-zero
            }
        }

        let mut merged: HashMap<usize, Fr> = HashMap::new();
        for (wire, coeff) in &self.terms {
            let entry = merged.entry(*wire).or_insert_with(Fr::zero);
            *entry = entry.add(coeff);
        }
        self.terms = merged.into_iter()
            .filter(|(_, coeff)| !coeff.0.is_zero())
            .collect();
        self.terms.sort_by_key(|t| t.0);
        self.is_simplified = true;
    }

    pub fn is_constant(&self) -> Option<Fr> {
        if self.terms.is_empty() {
            return Some(Fr::zero());
        }
        
        // A linear combination is not a constant if it contains any variable wire (id > 0)
        let mut has_variables = false;
        for (wire, coeff) in &self.terms {
            if *wire != 0 && !coeff.0.is_zero() {
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
            if coeff.0 == BigUint::one() {
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

        nodes.push(WitnessOp::Const(BigUint::one()));
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
        let mut mul_by_output: HashMap<usize, (usize, usize)> = HashMap::new();
        for c in &self.constraints {
            if c.c.terms.len() == 1 && c.a.terms.len() == 1 && c.b.terms.len() == 1 {
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
        // combinations. The `w < out` guard keeps the recipe evaluable in wire
        // order: a constraint whose `A` or `B` mentions its own output cannot
        // be used to define it, and would otherwise read a zero and look solved.
        let mut lc_by_output: HashMap<usize, (LinearCombination, LinearCombination)> = HashMap::new();
        for c in &self.constraints {
            if c.c.terms.len() != 1 || c.c.terms[0].1 != Fr::one() {
                continue;
            }
            let out = c.c.terms[0].0;
            if out == 0 || mul_by_output.contains_key(&out) || lc_by_output.contains_key(&out) {
                continue;
            }
            if c.a.terms.iter().chain(c.b.terms.iter()).all(|(w, _)| *w < out) {
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

        let topological_order = (0..num_signals).map(SignalId).collect();

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

        // Apply selected target configuration
        self.active_field = active_field;
        self.active_scheme = active_scheme;
        let config = FieldConfig::get(self.active_field);
        ACTIVE_MODULUS.with(|m| {
            *m.borrow_mut() = config.p.clone();
        });

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

        // Note: we must pass &flat_items here so all nested functions/structs are in scope
        self.emit_circuit_entry(f, &flat_items)?;

        // Run optimization pass: dead-wire elimination & constraint reduction
        self.optimize_circuit();

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
                        let max_val = max_fr.0;
                        let bit_len = max_val.bit_len();
                        
                        // We decompose lc into bit_len bits:
                        // lc = sum_{i=0}^{bit_len-1} b_i * 2^i
                        // and for each b_i, b_i * b_i = b_i
                        let mut sum_lc = LinearCombination::zero();
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
                            
                            let mut factor = BigUint::one();
                            for _ in 0..i {
                                factor = factor.mul(&BigUint::from_u64(2));
                            }
                            sum_lc.add_term(bit_var, Fr::from_biguint(factor));
                        }
                        
                        // Constrain: sum_lc = lc
                        self.constraints.push(Constraint {
                            a: sum_lc,
                            b: LinearCombination::constant(Fr::one()),
                            c: lc.clone(),
                            span: Some(bounds_attr.span.clone()),
                        });

                        // Also decompose (max_val - lc) into bit_len bits to ensure lc <= max_val!
                        let mut diff_lc = LinearCombination::constant(Fr::from_biguint(max_val.clone()));
                        diff_lc.add_linear(&lc, Fr::from_u64(0).sub(&Fr::one()));
                        
                        let mut diff_sum_lc = LinearCombination::zero();
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
                            
                            let mut factor = BigUint::one();
                            for _ in 0..i {
                                factor = factor.mul(&BigUint::from_u64(2));
                            }
                            diff_sum_lc.add_term(bit_var, Fr::from_biguint(factor));
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
                let step_bi = step_val.0;
                let end_bi = end_const.0;

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

                while current.0 < end_bi {
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
                    current = Fr(current.0.add(&step_bi));
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
                    if !c.0.is_zero() {
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
                    if const_val.0.is_zero() {
                        return Ok(None); // Statically false condition, zero iterations
                    }
                    
                    let mut is_static_throughout = true;
                    for _iter in 0..max_iters {
                        self.emit_block(body, items)?;
                        let next_cond_lc = self.emit_expr(condition, items)?;
                        if let Some(next_c) = next_cond_lc.is_constant() {
                            if next_c.0.is_zero() {
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
                        // Division: left / right
                        // Constant divisor: scale by inverse
                        if let Some(rc) = right_lc.is_constant() {
                            let inv_rc = rc.try_inv().map_err(|e| format!("Line {}: {}", span.line, e))?;
                            return Ok(left_lc.scale(inv_rc));
                        }

                        // Dynamic divisor: w_out * right = left
                        let out_wire = self.new_wire("div_tmp");
                        self.add_constraint(Constraint {
                            a: LinearCombination::variable(out_wire),
                            b: right_lc,
                            c: left_lc,
                            span: Some(span.clone()),
                        });
                        Ok(LinearCombination::variable(out_wire))
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
                            let eq_val = if dc.0.is_zero() { Fr::one() } else { Fr::zero() };
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
                                BinaryOp::Lt => lc.0 < rc.0,
                                BinaryOp::Le => lc.0 <= rc.0,
                                BinaryOp::Gt => lc.0 > rc.0,
                                BinaryOp::Ge => lc.0 >= rc.0,
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
                    _ => Err(format!("Circuit target error: Operator {:?} is not natively supported in ZK field constraints. Line {}", op, span.line)),
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
                let index_val = index_const.0.to_decimal_string();
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

    /// One R1CS constraint `a * b = w` for a fresh wire `w`, with a witness
    /// recipe so the forward pass can compute it.
    ///
    /// The recipe is the point. R1CS allows `a` and `b` to be arbitrary linear
    /// combinations, which is what makes Poseidon cheap - `(x + C)^2` needs no
    /// wire for `x + C`. But `build_witness_ir`'s reconstruction scan only
    /// recognises single-term sides, so without the explicit recipe every one
    /// of these wires falls to back-propagation.
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
        self.witness_recipes
            .insert(wire, WitnessOp::MulLc(a.clone(), b.clone()));
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

        let hex = |s: &str| -> Result<Fr, String> {
            BigUint::from_hex_str(s).map(Fr::from_biguint)
        };
        let c: Vec<Fr> = POSEIDON_C_T3.iter().map(|s| hex(s)).collect::<Result<_, _>>()?;
        let s_sparse: Vec<Fr> = POSEIDON_S_T3.iter().map(|s| hex(s)).collect::<Result<_, _>>()?;
        let mut m = [[Fr::zero(), Fr::zero(), Fr::zero()], [Fr::zero(), Fr::zero(), Fr::zero()], [Fr::zero(), Fr::zero(), Fr::zero()]];
        let mut p = m.clone();
        for i in 0..T {
            for j in 0..T {
                m[i][j] = hex(POSEIDON_M_T3[i][j])?;
                p[i][j] = hex(POSEIDON_P_T3[i][j])?;
            }
        }

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
                    out[i].add_linear(s, mat[j][i].clone());
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
            s.add_constant(c[j].clone());
        }

        // First half of the full rounds, all but the last.
        for r in 0..half_f - 1 {
            let sig: Vec<_> = state.iter().map(|x| self.emit_poseidon_sbox(&x.clone())).collect::<Vec<_>>();
            let mut arked = sig;
            for (j, s) in arked.iter_mut().enumerate() {
                s.add_constant(c[(r + 1) * T + j].clone());
            }
            state = mix(&arked, &m);
        }

        // Last full round of the first half mixes with P, the change of basis
        // into the sparse partial-round representation.
        {
            let sig: Vec<_> = state.iter().map(|x| self.emit_poseidon_sbox(&x.clone())).collect::<Vec<_>>();
            let mut arked = sig;
            for (j, s) in arked.iter_mut().enumerate() {
                s.add_constant(c[half_f * T + j].clone());
            }
            state = mix(&arked, &p);
        }

        // Partial rounds: S-box on lane 0 only, mixed by the sparse matrices.
        let width = T * 2 - 1;
        for r in 0..r_p {
            let mut inp = state.clone();
            inp[0] = self.emit_poseidon_sbox(&state[0]);
            inp[0].add_constant(c[(half_f + 1) * T + r].clone());

            let mut out = vec![LinearCombination::zero(); T];
            for (i, s) in inp.iter().enumerate().take(T) {
                out[0].add_linear(s, s_sparse[width * r + i].clone());
            }
            out[0].simplify();
            for j in 1..T {
                out[j] = inp[j].clone();
                out[j].add_linear(&inp[0], s_sparse[width * r + T + j - 1].clone());
                out[j].simplify();
            }
            state = out;
        }

        // Second half of the full rounds, all but the last.
        for r in 0..half_f - 1 {
            let sig: Vec<_> = state.iter().map(|x| self.emit_poseidon_sbox(&x.clone())).collect::<Vec<_>>();
            let mut arked = sig;
            for (j, s) in arked.iter_mut().enumerate() {
                s.add_constant(c[(half_f + 1) * T + r_p + r * T + j].clone());
            }
            state = mix(&arked, &m);
        }

        // Final round has no Ark; MixLast takes column 0 only.
        let sig: Vec<_> = state.iter().map(|x| self.emit_poseidon_sbox(&x.clone())).collect::<Vec<_>>();
        let mut out = LinearCombination::zero();
        for (j, s) in sig.iter().enumerate().take(T) {
            out.add_linear(s, m[j][0].clone());
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

        let mut iteration = 0;
        loop {
            let mut replacements = HashMap::new();
            let mut seen: HashMap<u64, Vec<(LinearCombination, LinearCombination, usize)>> = HashMap::new();
            let mut duplicate_indices = HashSet::new();

            for (idx, c) in self.constraints.iter().enumerate() {
                // We only optimize constraints where C is a single intermediate wire: w_j
                // A wire is intermediate if it is not 0, not in public/private inputs, and not in outputs.
                if c.c.terms.len() == 1 {
                    let (wire_j, coeff) = (c.c.terms[0].0, &c.c.terms[0].1);
                    if coeff.0 == BigUint::one()
                        && wire_j != 0
                        && !self.public_inputs.contains(&wire_j)
                        && !self.private_inputs.contains(&wire_j)
                        && !self.outputs.contains(&wire_j)
                    {
                        // Compute commutative hash for (A, B) using symmetric addition of individual hashes
                        use std::hash::{Hash, Hasher};
                        use std::collections::hash_map::DefaultHasher;

                        let hash_a = {
                            let mut s = DefaultHasher::new();
                            c.a.hash(&mut s);
                            s.finish()
                        };
                        let hash_b = {
                            let mut s = DefaultHasher::new();
                            c.b.hash(&mut s);
                            s.finish()
                        };
                        let combined_hash = hash_a.wrapping_add(hash_b);

                        let mut found_wire = None;
                        if let Some(candidates) = seen.get(&combined_hash) {
                            for (seen_a, seen_b, seen_wire) in candidates {
                                if (c.a == *seen_a && c.b == *seen_b) || (c.a == *seen_b && c.b == *seen_a) {
                                    found_wire = Some(*seen_wire);
                                    break;
                                }
                            }
                        }

                        if let Some(wire_i) = found_wire {
                            replacements.insert(wire_j, wire_i);
                            duplicate_indices.insert(idx);
                        } else {
                            seen.entry(combined_hash)
                                .or_insert_with(Vec::new)
                                .push((c.a.clone(), c.b.clone(), wire_j));
                        }
                    }
                }
            }

            if replacements.is_empty() {
                break;
            }

            // Apply replacements to all remaining constraints
            let mut new_constraints = Vec::new();
            for (idx, mut c) in self.constraints.drain(..).enumerate() {
                if duplicate_indices.contains(&idx) {
                    continue; // Remove the duplicate constraint
                }

                // Helper to replace wires in a linear combination
                let replace_lc = |lc: &mut LinearCombination, reps: &HashMap<usize, usize>| {
                    let mut changed = false;
                    for term in &mut lc.terms {
                        if let Some(&new_w) = reps.get(&term.0) {
                            term.0 = new_w;
                            changed = true;
                        }
                    }
                    if changed {
                        lc.is_simplified = false;
                        lc.simplify();
                    }
                };

                replace_lc(&mut c.a, &replacements);
                replace_lc(&mut c.b, &replacements);
                replace_lc(&mut c.c, &replacements);
                new_constraints.push(c);
            }
            self.constraints = new_constraints;
            iteration += 1;
            if iteration > 10 {
                break; // Safety limit
            }
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

    pub fn write_r1cs_binary(&self, output_path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::BufWriter;
        use std::io::Write;

        let circuit = self.build_circuit();

        // 1. Determine the old-to-new wire mapping
        let mut old_to_new = HashMap::new();
        old_to_new.insert(0, 0); // Constant 1 is mapped to wire 0

        let mut next_new_id = 1;

        // Public outputs map next
        for &w in &circuit.outputs {
            if !old_to_new.contains_key(&w) {
                old_to_new.insert(w, next_new_id);
                next_new_id += 1;
            }
        }
        let n_pub_out = next_new_id - 1;

        // Public inputs map next
        for &w in &circuit.public_inputs {
            if !old_to_new.contains_key(&w) {
                old_to_new.insert(w, next_new_id);
                next_new_id += 1;
            }
        }
        let n_pub_in = next_new_id - 1 - n_pub_out;

        // Private inputs map next
        for &w in &circuit.private_inputs {
            if !old_to_new.contains_key(&w) {
                old_to_new.insert(w, next_new_id);
                next_new_id += 1;
            }
        }
        let n_prv_in = next_new_id - 1 - n_pub_out - n_pub_in;

        // Intermediate/Auxiliary wires map last
        let mut aux_wires = Vec::new();
        for w in 1..circuit.num_variables {
            if !old_to_new.contains_key(&w) {
                aux_wires.push(w);
            }
        }
        aux_wires.sort();
        for w in aux_wires {
            old_to_new.insert(w, next_new_id);
            next_new_id += 1;
        }

        // 2. Write binary .r1cs file
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);
        let encoder = R1csEncoder::new();
        encoder.encode_to_stream(&circuit, &old_to_new, n_pub_out, n_pub_in, n_prv_in, &mut writer)?;

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
        circuit: &Circuit,
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
        let mut constraints_buf = Vec::new();
        for c in &circuit.constraints {
            for lc in &[&c.a, &c.b, &c.c] {
                let mut remapped_terms: Vec<(u32, Vec<u8>)> = Vec::new();
                for &(old_wire, ref coeff) in &lc.terms {
                    let new_wire = *old_to_new.get(&old_wire).unwrap_or(&0) as u32;
                    let coeff_bytes = coeff.0.to_bytes_le(32);
                    remapped_terms.push((new_wire, coeff_bytes));
                }
                // Sort by new wire ID ascending
                remapped_terms.sort_by_key(|t| t.0);

                let n_terms = remapped_terms.len() as u32;
                constraints_buf.write_all(&n_terms.to_le_bytes())?;

                for (wire_id, val_bytes) in remapped_terms {
                    constraints_buf.write_all(&wire_id.to_le_bytes())?;
                    constraints_buf.write_all(&val_bytes)?;
                }
            }
        }
        self.write_section(writer, 2, &constraints_buf)?;

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
            ACTIVE_MODULUS.with(|m| {
                *m.borrow_mut() = config.p.clone();
            });
            
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
        let p = ACTIVE_MODULUS.with(|m| m.borrow().clone());
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
