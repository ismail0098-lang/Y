// ============================================================
//  Y — ZK Backend Field Arithmetic
//  zk_field.rs
//
//  `BigUint`  — arbitrary-precision, heap-backed. Parsing, one-off setup,
//               and the handful of places that genuinely need unbounded width.
//  `Fr`       — a scalar-field element: `[u64; 4]` in MONTGOMERY FORM, `Copy`,
//               stack-resident, zero allocations for add / sub / mul.
// ============================================================
//
// Why `Fr` is not a `BigUint`:
//
// It was one, and that single line was the whole cost of ZK compilation.
// Profiling a 1000-hash Poseidon chain (241 constraints each) measured 8.6 s in
// `emit_program`, of which only ~1.8 s (20%) was field arithmetic — and
// **356,381,447 allocations**, 1,479 per constraint, 12.69 GB churned. A BN254
// element is 254 bits, four `u64`s, and storing it as a heap `Vec<u32>` meant
// every element created, cloned, returned by value or moved into a `HashMap`
// hit the allocator: `Fr::mul` alone performed 22.2 allocations, `Fr::add` 6.4.
// At ~24 ns per malloc/free pair that is ~8.5 s — essentially the entire emit.
//
// The obvious conclusion from "a Poseidon circuit is pure field arithmetic" is
// that the fix is a faster multiply. It is not: an infinitely fast `Fr::mul`
// would still have left 6.8 s. The fix is to stop allocating. See
// `docs/zk_emit_profile.md`.
//
// Consequences that matter to callers:
//   - `Fr` is `Copy`. `Vec<(usize, Fr)>` is a flat POD vector with no drop glue.
//   - The inner limbs are PRIVATE and in Montgomery form. They are not the
//     number. Everything goes through the API, which is deliberate: a `.0` that
//     used to read the value would now silently read `value * R mod p`.
//   - Correctness is guarded by `tests/zk_poseidon_interop.rs` (four pinned
//     circomlib digests), `tests/zk_groth16_end_to_end.rs` (real Groth16 over
//     arkworks) and `tests/zk_integer_ops.rs`. A pinned digest must never be
//     "updated" to accommodate a change here — if one moves, the field is wrong.

#![allow(dead_code)]

use std::cell::{Cell, RefCell};

// ────────────────────────────────────────────────────────
// 1. BigUint — arbitrary precision, base 2^32
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
            // Historical behaviour: an underflowing `BigUint` subtraction wraps
            // modulo the active field. Kept because the setup paths rely on it.
            // `try_active_modulus` rather than `active_modulus` because this can
            // be reached from inside field-parameter initialisation, where the
            // thread-local is still being constructed.
            if let Some(mod_val) = try_active_modulus() {
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
                let prod =
                    (self.digits[i] as u64) * (other.digits[j] as u64) + (digits[i + j] as u64) + carry;
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
        let last_idx = self.effective_len() - 1;
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

    /// The low 256 bits, as four little-endian `u64` limbs.
    pub fn to_limbs4(&self) -> [u64; 4] {
        let mut out = [0u64; 4];
        for (i, out_limb) in out.iter_mut().enumerate() {
            let lo = self.digits.get(2 * i).copied().unwrap_or(0) as u64;
            let hi = self.digits.get(2 * i + 1).copied().unwrap_or(0) as u64;
            *out_limb = lo | (hi << 32);
        }
        out
    }

    pub fn from_limbs4(limbs: &[u64; 4]) -> Self {
        let mut digits = Vec::with_capacity(8);
        for l in limbs {
            digits.push(*l as u32);
            digits.push((l >> 32) as u32);
        }
        let mut res = Self { digits };
        res.trim();
        res
    }
}

// ────────────────────────────────────────────────────────
// 2. 256-bit primitives
// ────────────────────────────────────────────────────────

const N: usize = 4;

#[inline(always)]
fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let r = (a as u128) + (b as u128) + (carry as u128);
    (r as u64, (r >> 64) as u64)
}

#[inline(always)]
fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let r = (a as u128).wrapping_sub(b as u128).wrapping_sub(borrow as u128);
    (r as u64, ((r >> 64) as u64) & 1)
}

/// `a + b*c + carry`, returning `(low, carry)`.
#[inline(always)]
fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let r = (a as u128) + (b as u128) * (c as u128) + (carry as u128);
    (r as u64, (r >> 64) as u64)
}

#[inline(always)]
fn cmp_limbs(a: &[u64; N], b: &[u64; N]) -> std::cmp::Ordering {
    for i in (0..N).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    std::cmp::Ordering::Equal
}

#[inline(always)]
fn sub_limbs(a: &[u64; N], b: &[u64; N]) -> ([u64; N], u64) {
    let mut r = [0u64; N];
    let mut borrow = 0u64;
    for i in 0..N {
        let (v, br) = sbb(a[i], b[i], borrow);
        r[i] = v;
        borrow = br;
    }
    (r, borrow)
}

#[inline(always)]
fn add_limbs(a: &[u64; N], b: &[u64; N]) -> ([u64; N], u64) {
    let mut r = [0u64; N];
    let mut carry = 0u64;
    for i in 0..N {
        let (v, c) = adc(a[i], b[i], carry);
        r[i] = v;
        carry = c;
    }
    (r, carry)
}

/// Montgomery product: `a * b * R^-1 mod p`, with `R = 2^256`.
///
/// Separated Operand Scanning. `inv` is `-p^-1 mod 2^64`. Requires `p` odd
/// (every scalar field modulus is prime) and `p < 2^255`, which
/// `FieldParams::new` checks so the single conditional subtraction below is
/// provably enough: the intermediate `u` is `< 2p`.
#[inline(always)]
fn mont_mul(a: &[u64; N], b: &[u64; N], p: &[u64; N], inv: u64) -> [u64; N] {
    let mut t = [0u64; 2 * N + 1];

    for i in 0..N {
        let mut carry = 0u64;
        for j in 0..N {
            let (v, c) = mac(t[i + j], a[j], b[i], carry);
            t[i + j] = v;
            carry = c;
        }
        let (v, c) = adc(t[i + N], carry, 0);
        t[i + N] = v;
        t[i + N + 1] = c;
    }

    for i in 0..N {
        let m = t[i].wrapping_mul(inv);
        let mut carry = 0u64;
        for j in 0..N {
            let (v, c) = mac(t[i + j], m, p[j], carry);
            t[i + j] = v;
            carry = c;
        }
        // Koc's SOS calls this step ADD(t[i+s], C), and ADD *propagates*: the
        // carry out of `t[i+N+1] + c` must keep rippling upward. Stopping after
        // one word drops it, which is wrong only for operand pairs that happen
        // to overflow that word — under Pallas, `inv(2)` and `inv(3)` came back
        // wrong while `inv(5)` was right. A data-dependent 1-in-2^64-ish bug is
        // exactly what a differential test over one field will not find.
        let (v, mut c) = adc(t[i + N], carry, 0);
        t[i + N] = v;
        let mut k = i + N + 1;
        while c != 0 && k <= 2 * N {
            let (v2, c2) = adc(t[k], c, 0);
            t[k] = v2;
            c = c2;
            k += 1;
        }
    }

    let mut u = [0u64; N];
    u.copy_from_slice(&t[N..2 * N]);

    // u < 2p, so at most one subtraction. t[2N] is the carry word.
    if t[2 * N] != 0 || cmp_limbs(&u, p) != std::cmp::Ordering::Less {
        let (r, _) = sub_limbs(&u, p);
        return r;
    }
    u
}

// ────────────────────────────────────────────────────────
// 3. Active field parameters
// ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct FieldParams {
    /// The modulus, arbitrary precision. Kept for the writers and for callers
    /// that need it as a number rather than as field parameters.
    pub p_big: BigUint,
    pub p: [u64; N],
    /// `-p^-1 mod 2^64`
    pub inv: u64,
    /// `R mod p` — the Montgomery representation of 1.
    pub r1: [u64; N],
    /// `R^2 mod p` — multiplying by this converts into Montgomery form.
    pub r2: [u64; N],
}

impl FieldParams {
    /// Derive the Montgomery constants for `p`.
    ///
    /// Runs `BigUint` long division twice; it happens once per field, not per
    /// operation. Must not call anything that reads the active-modulus
    /// thread-local, because it runs while that thread-local is initialising.
    pub fn new(p_big: &BigUint) -> Self {
        assert!(
            p_big.bit_len() <= 255,
            "scalar field modulus must be < 2^255 (got {} bits); the Montgomery \
             reduction's single conditional subtraction assumes 2p fits in 256 bits",
            p_big.bit_len()
        );
        assert!(
            p_big.get_bit(0),
            "scalar field modulus must be odd for Montgomery multiplication"
        );

        let p = p_big.to_limbs4();

        // -p^-1 mod 2^64, by Newton iteration on the low limb.
        let mut inv = 1u64;
        for _ in 0..63 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(p[0]);
        }
        let inv = inv.wrapping_neg();

        // R = 2^256, R^2 = 2^512.
        let mut r_num = BigUint { digits: vec![0u32; 9] };
        r_num.digits[8] = 1;
        let r1 = r_num.div_mod(p_big).1.to_limbs4();

        let mut r2_num = BigUint { digits: vec![0u32; 17] };
        r2_num.digits[16] = 1;
        let r2 = r2_num.div_mod(p_big).1.to_limbs4();

        let params = FieldParams { p_big: p_big.clone(), p, inv, r1, r2 };

        // A composite modulus does not give a field, and an R1CS emitted over
        // Z/nZ for composite n proves nothing: inverses stop existing, so
        // is-zero gadgets and division become unsatisfiable or forgeable
        // depending on the factor. Nothing downstream can detect it — the
        // constraints still emit, the writers still write, and Groth16 over the
        // wrong ring is simply broken cryptography.
        //
        // This is not hypothetical. `ScalarField::Vesta`'s modulus in this file
        // WAS composite: a corrupted transcription of Vesta's base field that
        // shared the first 21 hex digits with the real one. It survived because
        // the old `Fr::inv` was extended Euclid, which happily inverts anything
        // coprime to the modulus and so produced plausible values.
        assert!(
            params.is_probable_prime(),
            "scalar field modulus is not prime: {}\n  \
             An R1CS over a composite modulus is not a proof system - inverses \
             do not exist and no gadget that needs one is sound.",
            p_big.to_decimal_string()
        );

        params
    }

    /// Miller-Rabin over the first twelve prime bases.
    ///
    /// Run once per field switch (~100 us), never per operation. Twelve bases
    /// is deterministic below 3.3e24 and leaves an error under 4^-12 for the
    /// 254-bit moduli here — and the failure it exists to catch is a mistyped
    /// constant, not an adversarially constructed pseudoprime.
    pub fn is_probable_prime(&self) -> bool {
        let n = self.p;
        if n == [2, 0, 0, 0] {
            return true;
        }
        if n[0] & 1 == 0 || cmp_limbs(&n, &[3, 0, 0, 0]) == std::cmp::Ordering::Less {
            return false;
        }

        // n - 1 = d * 2^s
        let (mut d, _) = sub_limbs(&n, &[1, 0, 0, 0]);
        let mut s = 0u32;
        while d[0] & 1 == 0 {
            d = shr1(&d);
            s += 1;
        }

        let neg_one_mont = sub_limbs(&n, &self.r1).0;

        'base: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            if cmp_limbs(&[a, 0, 0, 0], &n) != std::cmp::Ordering::Less {
                continue;
            }
            let a_mont = mont_mul(&[a, 0, 0, 0], &self.r2, &n, self.inv);
            let mut x = self.mont_pow(&a_mont, &d);
            if x == self.r1 || x == neg_one_mont {
                continue;
            }
            for _ in 0..s.saturating_sub(1) {
                x = mont_mul(&x, &x, &n, self.inv);
                if x == neg_one_mont {
                    continue 'base;
                }
            }
            return false;
        }
        true
    }

    /// `base^exp` with both `base` and the result in Montgomery form.
    fn mont_pow(&self, base: &[u64; N], exp: &[u64; N]) -> [u64; N] {
        let mut acc = self.r1;
        let mut started = false;
        for i in (0..N).rev() {
            for b in (0..64).rev() {
                if started {
                    acc = mont_mul(&acc, &acc, &self.p, self.inv);
                }
                if (exp[i] >> b) & 1 == 1 {
                    acc = if started {
                        mont_mul(&acc, base, &self.p, self.inv)
                    } else {
                        *base
                    };
                    started = true;
                }
            }
        }
        acc
    }
}

#[inline]
fn shr1(a: &[u64; N]) -> [u64; N] {
    let mut r = [0u64; N];
    let mut carry = 0u64;
    for i in (0..N).rev() {
        r[i] = (a[i] >> 1) | (carry << 63);
        carry = a[i] & 1;
    }
    r
}

pub const BN254_FR_MODULUS: &str =
    "21888242871839275222246405745257275088548364400416034343698204186575808495617";

thread_local! {
    static FIELD: RefCell<FieldParams> =
        RefCell::new(FieldParams::new(&BigUint::from_str(BN254_FR_MODULUS)));
}

/// Switch the active scalar field. Recomputes the Montgomery constants.
///
/// Every `Fr` already in existence was built against the OLD field and is not
/// reinterpreted; callers switch fields before constructing elements, which is
/// what `FieldConfig::get` and `emit_program` do.
pub fn set_active_modulus(p: &BigUint) {
    let params = FieldParams::new(p);
    FIELD.with(|f| *f.borrow_mut() = params);
}

pub fn active_modulus() -> BigUint {
    FIELD.with(|f| f.borrow().p_big.clone())
}

/// `active_modulus`, but `None` rather than a panic when the thread-local is
/// unavailable or mid-initialisation. See `BigUint::sub`.
fn try_active_modulus() -> Option<BigUint> {
    FIELD
        .try_with(|f| f.try_borrow().ok().map(|f| f.p_big.clone()))
        .ok()
        .flatten()
}

pub fn with_field_params<R, F: FnOnce(&FieldParams) -> R>(f: F) -> R {
    FIELD.with(|c| f(&c.borrow()))
}

// ────────────────────────────────────────────────────────
// 4. Field-operation counters
// ────────────────────────────────────────────────────────

// Counts of `Fr::mul` / `Fr::add`, for attributing ZK emission cost.
//
// Timing alone cannot separate "the field multiply is slow" from "we perform
// too many field multiplies", and those have completely different fixes.
//
// Thread-local `Cell`s rather than atomics: the ZK pipeline is single-threaded
// and a relaxed `fetch_add` per operation was affordable against an 8.6 s emit
// but is not against a sub-second one - the counters would have become a
// measurable fraction of the thing they measure.
thread_local! {
    static FR_MUL_COUNT: Cell<u64> = const { Cell::new(0) };
    static FR_ADD_COUNT: Cell<u64> = const { Cell::new(0) };
}

#[inline(always)]
fn count_mul() {
    FR_MUL_COUNT.with(|c| c.set(c.get().wrapping_add(1)));
}

#[inline(always)]
fn count_add() {
    FR_ADD_COUNT.with(|c| c.set(c.get().wrapping_add(1)));
}

/// `(multiplies, adds)` performed so far on this thread.
pub fn field_op_counts() -> (u64, u64) {
    (FR_MUL_COUNT.with(|c| c.get()), FR_ADD_COUNT.with(|c| c.get()))
}

// ────────────────────────────────────────────────────────
// 5. Fr — a scalar field element
// ────────────────────────────────────────────────────────

/// An element of the active scalar field, stored as `value * R mod p` with
/// `R = 2^256`.
///
/// The limbs are private, and that is load-bearing. In Montgomery form the
/// stored limbs are NOT the number; a caller reaching past the API to read them
/// would get `value * R mod p` and no type error.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Fr([u64; N]);

impl Fr {
    // ---- construction ----

    pub fn zero() -> Self {
        Fr([0u64; N])
    }

    pub fn one() -> Self {
        Fr(with_field_params(|f| f.r1))
    }

    pub fn from_u64(val: u64) -> Self {
        Self::from_limbs_reduce([val, 0, 0, 0])
    }

    pub fn from_biguint(bi: BigUint) -> Self {
        Self::from_biguint_ref(&bi)
    }

    pub fn from_biguint_ref(bi: &BigUint) -> Self {
        with_field_params(|f| {
            // Reduce with long division only when the value cannot fit in 256
            // bits; otherwise the conditional-subtraction loop below is enough
            // and does not allocate.
            let limbs = if bi.effective_len() > 8 {
                bi.div_mod(&f.p_big).1.to_limbs4()
            } else {
                bi.to_limbs4()
            };
            Fr(mont_mul(&reduce_once(limbs, &f.p), &f.r2, &f.p, f.inv))
        })
    }

    /// From a canonical (non-Montgomery) 256-bit value, reducing mod p.
    pub fn from_limbs_reduce(limbs: [u64; N]) -> Self {
        with_field_params(|f| Fr(mont_mul(&reduce_once(limbs, &f.p), &f.r2, &f.p, f.inv)))
    }

    // ---- accessors ----

    /// The canonical value, as four little-endian `u64` limbs.
    pub fn to_limbs(&self) -> [u64; N] {
        with_field_params(|f| mont_mul(&self.0, &[1, 0, 0, 0], &f.p, f.inv))
    }

    pub fn to_biguint(&self) -> BigUint {
        BigUint::from_limbs4(&self.to_limbs())
    }

    /// The Montgomery limbs. For serialisation of the *internal* form only —
    /// almost every caller wants `to_limbs`.
    pub fn mont_limbs(&self) -> [u64; N] {
        self.0
    }

    pub fn is_zero(&self) -> bool {
        self.0 == [0u64; N]
    }

    pub fn is_one(&self) -> bool {
        *self == Fr::one()
    }

    /// Bit `i` of the canonical value, LSB-first.
    pub fn get_bit(&self, i: usize) -> bool {
        if i >= 256 {
            return false;
        }
        (self.to_limbs()[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Number of significant bits in the canonical value; 0 for zero.
    pub fn bit_len(&self) -> usize {
        let l = self.to_limbs();
        for i in (0..N).rev() {
            if l[i] != 0 {
                return i * 64 + (64 - l[i].leading_zeros() as usize);
            }
        }
        0
    }

    /// The canonical value as a `u64`, when it fits.
    pub fn to_u64(&self) -> Option<u64> {
        let l = self.to_limbs();
        if l[1] == 0 && l[2] == 0 && l[3] == 0 {
            Some(l[0])
        } else {
            None
        }
    }

    // ---- arithmetic ----

    pub fn add(&self, other: &Self) -> Self {
        count_add();
        with_field_params(|f| {
            let (sum, carry) = add_limbs(&self.0, &other.0);
            if carry != 0 || cmp_limbs(&sum, &f.p) != std::cmp::Ordering::Less {
                Fr(sub_limbs(&sum, &f.p).0)
            } else {
                Fr(sum)
            }
        })
    }

    pub fn sub(&self, other: &Self) -> Self {
        with_field_params(|f| {
            let (diff, borrow) = sub_limbs(&self.0, &other.0);
            if borrow != 0 {
                Fr(add_limbs(&diff, &f.p).0)
            } else {
                Fr(diff)
            }
        })
    }

    pub fn neg(&self) -> Self {
        Fr::zero().sub(self)
    }

    pub fn mul(&self, other: &Self) -> Self {
        count_mul();
        with_field_params(|f| Fr(mont_mul(&self.0, &other.0, &f.p, f.inv)))
    }

    pub fn square(&self) -> Self {
        self.mul(self)
    }

    pub fn double(&self) -> Self {
        self.add(self)
    }

    /// `self^exp`, exponent given as canonical little-endian limbs.
    pub fn pow_limbs(&self, exp: &[u64; N]) -> Self {
        with_field_params(|f| Fr(f.mont_pow(&self.0, exp)))
    }

    pub fn try_inv(&self) -> Result<Self, String> {
        if self.is_zero() {
            return Err("error[Z0040]: division by zero: finite field element zero has no modular inverse during host witness generation in @hint block".to_string());
        }
        Ok(self.inv())
    }

    /// Multiplicative inverse, by Fermat's little theorem: `a^(p-2)`.
    ///
    /// ~380 Montgomery multiplies, a few microseconds. It replaced an extended
    /// Euclid over `BigUint` — a bit-at-a-time `div_mod` per iteration, so
    /// milliseconds and thousands of allocations. Fermat is valid because every
    /// scalar field modulus here is prime, which is also what makes the field a
    /// field; `FieldParams::new` rejects an even modulus outright.
    pub fn inv(&self) -> Self {
        if self.is_zero() {
            panic!("Zero has no modular inverse");
        }
        let exp = with_field_params(|f| {
            let (mut e, borrow) = sub_limbs(&f.p, &[2, 0, 0, 0]);
            debug_assert_eq!(borrow, 0);
            if borrow != 0 {
                e = [0; N];
            }
            e
        });
        self.pow_limbs(&exp)
    }

    // ---- integer (non-field) operations on the canonical value ----

    /// Integer quotient and remainder of the canonical values.
    ///
    /// This is INTEGER division, not field division, and the distinction is the
    /// whole reason `IntDivLc` exists: `7 / 2` is 3, not `(p+7)/2`.
    pub fn int_div_rem(&self, other: &Self) -> (Self, Self) {
        let a = self.to_limbs();
        let b = other.to_limbs();
        if b == [0u64; N] {
            panic!("Fr::int_div_rem: division by zero");
        }
        let (q, r) = div_rem_u256(&a, &b);
        (Fr::from_limbs_reduce(q), Fr::from_limbs_reduce(r))
    }

    // ---- serialisation ----

    pub fn to_bytes_le(&self, byte_len: usize) -> Vec<u8> {
        let l = self.to_limbs();
        let mut bytes = Vec::with_capacity(byte_len.max(32));
        for limb in l.iter() {
            bytes.extend_from_slice(&limb.to_le_bytes());
        }
        bytes.resize(byte_len, 0);
        bytes
    }

    /// Little-endian canonical bytes into a caller-owned buffer.
    ///
    /// The allocating `to_bytes_le` is fine for a header field, but the `.r1cs`
    /// constraints section calls this once per TERM - 6.6 M times on a 237 k
    /// constraint Poseidon circuit - and a 32-byte `Vec` per term was the single
    /// largest source of allocations left in the whole ZK pipeline.
    pub fn write_bytes_le(&self, out: &mut [u8; 32]) {
        for (i, limb) in self.to_limbs().iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
    }

    pub fn to_decimal_string(&self) -> String {
        let mut v = self.to_limbs();
        if v == [0u64; N] {
            return "0".to_string();
        }
        // 10^19 is the largest power of ten fitting in a u64.
        const CHUNK: u64 = 10_000_000_000_000_000_000;
        let mut chunks: Vec<u64> = Vec::with_capacity(5);
        while v != [0u64; N] {
            let (q, r) = div_rem_small_u256(&v, CHUNK);
            v = q;
            chunks.push(r);
        }
        let mut s = chunks.last().unwrap().to_string();
        for chunk in chunks.iter().rev().skip(1) {
            s.push_str(&format!("{:019}", chunk));
        }
        s
    }

    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        self.to_decimal_string()
    }

    /// The active modulus. Kept as a `BigUint` because its consumers are the
    /// `.r1cs`/`.wtns` header writers and the arkworks cross-check.
    pub fn modulus() -> BigUint {
        active_modulus()
    }
}

impl PartialOrd for Fr {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Ordering is by CANONICAL value, not by the stored Montgomery limbs.
///
/// The two disagree on almost every pair, and the emitter uses this to
/// constant-fold `<`, `<=`, `>`, `>=` and to drive unrolled loop bounds — so
/// ordering the internal representation instead would fold comparisons to the
/// wrong answer and silently emit a circuit proving a false statement.
impl Ord for Fr {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        cmp_limbs(&self.to_limbs(), &other.to_limbs())
    }
}

impl std::fmt::Display for Fr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_decimal_string())
    }
}

/// Bring an arbitrary 256-bit value below `p` by repeated subtraction.
///
/// `2^256 / p < 6` for every field here, so this is at most five iterations and
/// never a long division.
#[inline]
fn reduce_once(mut limbs: [u64; N], p: &[u64; N]) -> [u64; N] {
    while cmp_limbs(&limbs, p) != std::cmp::Ordering::Less {
        limbs = sub_limbs(&limbs, p).0;
    }
    limbs
}

/// Schoolbook binary long division on 256-bit values.
fn div_rem_u256(a: &[u64; N], b: &[u64; N]) -> ([u64; N], [u64; N]) {
    let mut q = [0u64; N];
    let mut r = [0u64; N];
    for i in (0..256).rev() {
        // r <<= 1
        let mut carry = 0u64;
        for limb in r.iter_mut() {
            let next = *limb >> 63;
            *limb = (*limb << 1) | carry;
            carry = next;
        }
        // r |= bit i of a
        if (a[i / 64] >> (i % 64)) & 1 == 1 {
            r[0] |= 1;
        }
        if cmp_limbs(&r, b) != std::cmp::Ordering::Less {
            r = sub_limbs(&r, b).0;
            q[i / 64] |= 1 << (i % 64);
        }
    }
    (q, r)
}

/// Divide a 256-bit value by a `u64`, one pass.
#[inline]
fn div_rem_small_u256(a: &[u64; N], d: u64) -> ([u64; N], u64) {
    let mut q = [0u64; N];
    let mut rem: u128 = 0;
    for i in (0..N).rev() {
        let cur = (rem << 64) | a[i] as u128;
        q[i] = (cur / d as u128) as u64;
        rem = cur % d as u128;
    }
    (q, rem as u64)
}

// ────────────────────────────────────────────────────────
// 6. Tests
// ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bn254_p() -> BigUint {
        BigUint::from_str(BN254_FR_MODULUS)
    }

    /// Every modulus the backend can select. Tests run over ALL of them, not
    /// just BN254: the lost-carry bug in `mont_mul`'s reduction step was
    /// invisible under BN254 and wrong under Pallas, and a differential test
    /// pinned to the default field would have shipped it.
    const SUPPORTED_MODULI: [&str; 4] = [
        BN254_FR_MODULUS,
        "52435875175126190479447740508185965837690552500527637822603658699938581184513",
        "28948022309329048855892746252171976963363056481941560715954676764349967630337",
        "28948022309329048855892746252171976963363056481941647379679742748393362948097",
    ];

    /// Run `f` under each supported field, restoring the caller's field after.
    fn for_each_field(mut f: impl FnMut(&BigUint)) {
        let original = active_modulus();
        for m in SUPPORTED_MODULI {
            let p = BigUint::from_str(m);
            set_active_modulus(&p);
            f(&p);
        }
        set_active_modulus(&original);
    }

    #[test]
    fn round_trips_through_montgomery_form() {
        for v in [0u64, 1, 2, 7, u32::MAX as u64, u64::MAX] {
            let f = Fr::from_u64(v);
            assert_eq!(f.to_u64(), Some(v), "{} did not round-trip", v);
            assert_eq!(f.to_decimal_string(), v.to_string());
        }
    }

    /// The reference the whole representation has to agree with: the same
    /// arithmetic done in `BigUint` with an explicit `mod p`.
    ///
    /// Montgomery form is exactly the kind of change that produces
    /// self-consistent wrong answers — every internal check passes because
    /// everything is wrong the same way. Differential testing against the slow
    /// path is what catches that.
    #[test]
    fn agrees_with_biguint_reference() {
        for_each_field(|p| agrees_with_biguint_reference_over(p));
    }

    fn agrees_with_biguint_reference_over(p: &BigUint) {
        let mut x = 0x9E3779B97F4A7C15u64;
        for _ in 0..300 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let y = x.wrapping_mul(0xBF58476D1CE4E5B9);

            let a_big = BigUint::from_u64(x).mul(&BigUint::from_u64(y));
            let b_big = BigUint::from_u64(y).mul(&BigUint::from_u64(x ^ 0xdeadbeef));
            let a = Fr::from_biguint(a_big.clone());
            let b = Fr::from_biguint(b_big.clone());

            let want_a = a_big.div_mod(p).1;
            assert_eq!(a.to_biguint(), want_a, "reduction");

            let want_sum = a_big.add(&b_big).div_mod(p).1;
            assert_eq!(a.add(&b).to_biguint(), want_sum, "add");

            let want_prod = a_big.mul(&b_big).div_mod(p).1;
            assert_eq!(a.mul(&b).to_biguint(), want_prod, "mul");

            let want_diff = if a_big.div_mod(p).1 >= b_big.div_mod(p).1 {
                a_big.div_mod(p).1.sub(&b_big.div_mod(p).1)
            } else {
                a_big.div_mod(p).1.add(p).sub(&b_big.div_mod(p).1)
            };
            assert_eq!(a.sub(&b).to_biguint(), want_diff, "sub");
        }
    }

    #[test]
    fn inverse_is_a_true_inverse() {
        for_each_field(|p| {
            // The small elements first, and deliberately. `inv(2)` and `inv(3)`
            // were the two that exposed the dropped carry; a test seeded only
            // from a PRNG missed them because their inverses are the values
            // with the most structure - (p+1)/2, (2p+1)/3 - and so the most
            // carries.
            for v in 1u64..=64 {
                let a = Fr::from_u64(v);
                assert_eq!(a.mul(&a.inv()), Fr::one(), "{} * {}^-1 != 1 mod {}", v, v, p.to_decimal_string());
            }
            let mut x = 12345u64;
            for _ in 0..64 {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                let a = Fr::from_u64(x | 1);
                assert_eq!(a.mul(&a.inv()), Fr::one(), "a * a^-1 != 1 for {}", x);
            }
            assert!(Fr::zero().try_inv().is_err());
        });
    }

    /// Ordering must follow the canonical value. If it followed the stored
    /// Montgomery limbs the emitter would constant-fold comparisons wrongly.
    #[test]
    fn ordering_is_by_canonical_value() {
        assert!(Fr::from_u64(3) < Fr::from_u64(5));
        assert!(Fr::from_u64(5) > Fr::from_u64(3));
        assert!(Fr::from_u64(5) >= Fr::from_u64(5));
        assert!(Fr::zero() < Fr::one());
        let neg_one = Fr::zero().sub(&Fr::one());
        assert!(neg_one > Fr::from_u64(u64::MAX), "p-1 is larger than any u64");
    }

    #[test]
    fn integer_division_is_integer_not_field() {
        let (q, r) = Fr::from_u64(7).int_div_rem(&Fr::from_u64(2));
        assert_eq!(q.to_u64(), Some(3));
        assert_eq!(r.to_u64(), Some(1));
        // Field division would give (p+7)/2, a 254-bit number.
        assert_ne!(q, Fr::from_u64(7).mul(&Fr::from_u64(2).inv()));
    }

    #[test]
    fn write_bytes_le_matches_to_bytes_le() {
        for_each_field(|_| {
            let mut x = 0x9E3779B97F4A7C15u64;
            for _ in 0..64 {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = Fr::from_u64(x).mul(&Fr::from_u64(x ^ 0xdead));
                let mut buf = [0u8; 32];
                v.write_bytes_le(&mut buf);
                assert_eq!(buf.to_vec(), v.to_bytes_le(32));
            }
        });
    }

    #[test]
    fn decimal_and_bytes_match_biguint() {
        let p = bn254_p();
        let neg_one = Fr::zero().sub(&Fr::one());
        assert_eq!(neg_one.to_decimal_string(), p.sub(&BigUint::one()).to_decimal_string());
        assert_eq!(neg_one.to_bytes_le(32), p.sub(&BigUint::one()).to_bytes_le(32));
        assert_eq!(Fr::zero().to_bytes_le(32), vec![0u8; 32]);
    }

    #[test]
    fn bit_access_matches_biguint() {
        let v = Fr::from_biguint(BigUint::from_hex_str("0x1234567890abcdef1122334455667788").unwrap());
        let b = v.to_biguint();
        for i in 0..256 {
            assert_eq!(v.get_bit(i), b.get_bit(i), "bit {}", i);
        }
        assert_eq!(v.bit_len(), b.bit_len());
    }

    /// Every field the backend offers must produce usable Montgomery constants.
    #[test]
    fn all_supported_moduli_are_montgomery_ready() {
        let moduli = [
            BN254_FR_MODULUS,
            "52435875175126190479447740508185965837690552500527637822603658699938581184513",
            "28948022309329048855892746252171976963363056481941560715954676764349967630337",
            "28948022309329048855892746252171976963363056481941647379679742748393362948097",
        ];
        let original = active_modulus();
        for m in moduli {
            let p = BigUint::from_str(m);
            set_active_modulus(&p);
            let neg_one = Fr::zero().sub(&Fr::one());
            assert_eq!(neg_one.add(&Fr::one()), Fr::zero(), "wrap failed for {}", m);
            assert_eq!(neg_one.to_biguint(), p.sub(&BigUint::one()));
            let a = Fr::from_u64(0xdeadbeefcafe);
            assert_eq!(a.mul(&a.inv()), Fr::one(), "inverse failed for {}", m);
        }
        set_active_modulus(&original);
    }
}
