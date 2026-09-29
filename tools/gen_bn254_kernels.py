#!/usr/bin/env python3
"""Generates the BN254 field kernels in Y (`.ysu`).

Why generated rather than written by hand: the Y PTX backend has no local
arrays, and `let` binds a name to exactly one register. An eight-limb value
therefore has to be eight named scalars, and any loop over limbs has to be
unrolled so every limb index is a literal. `tests/bn254_fr_mul.ysu` is the
hand-written version that keeps its accumulator in per-thread GLOBAL scratch
instead; it is readable, it is the reference for what this computes, and it
measures 1.03x a single CPU core because every one of the ~128 multiply-
accumulate steps pays a global load and a global store.

Everything here is the same CIOS algorithm with the limbs held in registers
and the modulus as immediates. Regenerate with:

    python3 tools/gen_bn254_kernels.py

and re-run `cargo test --release --features zk --test zk_gpu_field`, which
checks both kernels against `src/zk_field.rs` on the device.
"""

# BN254 has two 254-bit primes and they are NOT interchangeable: Fr is the
# scalar field (the order of G1, what a witness lives in), Fq is the base
# field (what a point's coordinates live in). MSM needs Fq; the NTT needs Fr.
# Every routine below reads the module-level P32/NP, and `use_field` switches
# them, so one CIOS implementation serves both.
FR = 21888242871839275222246405745257275088548364400416034343698204186575808495617
FQ = 21888242871839275222246405745257275088696311157297823662689037894645226208583
S = 8   # limbs


# Y reserves U8/U16/U32/U64/I8/... as TYPE keywords, so a generated temporary
# called `U1` produces `U16` at limb 6 and the parser rejects it as a `let`
# name. Any new temporary prefix must avoid colliding once a limb index is
# appended - hence `UA`/`UB` rather than `U1`/`U2`.
RESERVED = {f"{p}{w}" for p in "UI" for w in (8, 16, 32, 64)}


def check_names(names):
    bad = [f"{v}{j}" for v in names for j in range(S) if f"{v}{j}" in RESERVED]
    assert not bad, f"generated variable names collide with type keywords: {bad}"


def use_field(p):
    global P, P32, NP
    P = p
    P32 = [(p >> (32 * i)) & 0xFFFFFFFF for i in range(S)]
    NP = (-pow(p, -1, 1 << 32)) % (1 << 32)   # -p^-1 mod 2^32
    assert (p * NP + 1) % (1 << 32) == 0


use_field(FR)


# ── Carry-chained field arithmetic ────────────────────────────────────────
#
# `CARRY = True` selects the versions built on PTX's condition-code
# instructions (`mad.lo.cc` / `madc.hi.cc` / `add.cc` / `subc`); False selects
# the original ones, which express every carry as the high half of an explicit
# 64-bit value. Both are kept because the only honest way to state what the
# rewrite bought is to measure the same kernel both ways.
#
# The reason to do it at all is NCU. With the stage-fused NTT no longer
# DRAM-bound, its SASS is 2,400 `IMAD.WIDE.U32` against ~10,500 instructions of
# carry bookkeeping -- the multiplies are 18% of the stream. `mul_wide_u32(a,b)
# + t + c` costs an IMAD.WIDE, an IADD3/IADD3.X pair and a shift-and-truncate,
# where `mad.lo.cc` + `madc.hi.cc` is two instructions total.
#
# THE ORDER OF THESE STATEMENTS IS SEMANTIC. Each chain talks to the next link
# through the condition code, which appears in no operand. Nothing may be
# emitted between links that writes CC -- a `mov` for an immediate or a load
# does not, another `.cc` instruction does.
CARRY = True

# Every limb value is 32 bits, so `-x` is written as this minus x when a mask
# is needed. Spelled out because `~` is not in the language.
U32_MAX = 0xFFFFFFFF


def _mac_row(T, a, bi, ind):
    """`T += a * bi`, as the two-pass carry-chained accumulate.

    `T` names S+2 accumulator limbs; `a` is a variable prefix and `bi` a single
    limb name (or an immediate). T[S+1] must be zero on entry.

    Two chains, not one. The `lo` pass adds `lo(a_j * bi)` into `T[j]`, the
    `hi` pass adds `hi(a_j * bi)` into `T[j+1]` -- the same total as the
    sequential 64-bit form, but each pass is a single hardware carry chain
    instead of S separate 64-bit accumulates.
    """
    L = [f"{ind}// t += a * b_i  (lo pass, then hi pass one limb up)"]
    L.append(f"{ind}{T[0]} = mad_lo_cc_u32({a}0, {bi}, {T[0]});")
    for j in range(1, S):
        L.append(f"{ind}{T[j]} = madc_lo_cc_u32({a}{j}, {bi}, {T[j]});")
    L.append(f"{ind}{T[S]} = addc_cc_u32({T[S]}, 0);")
    L.append(f"{ind}{T[S+1]} = addc_u32({T[S+1]}, 0);")
    L.append(f"{ind}{T[1]} = mad_hi_cc_u32({a}0, {bi}, {T[1]});")
    for j in range(1, S):
        L.append(f"{ind}{T[j+1]} = madc_hi_cc_u32({a}{j}, {bi}, {T[j+1]});")
    L.append(f"{ind}{T[S+1]} = addc_u32({T[S+1]}, 0);")
    return L


def _cond_sub(dst, T, ind):
    """`dst = T - p` if `T >= p` else `T`, over S limbs, branch-free.

    `subc_u32(0, 0)` is `0 - 0 - borrow`, i.e. 0 when T >= p and 0xFFFFFFFF
    when T < p -- the select mask falls straight out of the borrow chain
    instead of having to be rebuilt from a high half. Confirmed on the device,
    not assumed: see `tests/ptx_carry_chain.rs`.
    """
    L = [f"{ind}// t is below 2p; subtract p once if it is at least p."]
    L.append(f"{ind}d0 = sub_cc_u32({T[0]}, {P32[0]});")
    for j in range(1, S):
        L.append(f"{ind}d{j} = subc_cc_u32({T[j]}, {P32[j]});")
    L.append(f"{ind}mask = subc_u32(0, 0);")
    L.append(f"{ind}nmask = mask ^ {U32_MAX};")
    for j in range(S):
        L.append(f"{ind}{dst}{j} = (d{j} & nmask) | ({T[j]} & mask);")
    return L


def mont_mul_cc(dst, a, b, ind="    "):
    """CIOS with the carries in hardware. Same algorithm as `mont_mul_u64`.

    The per-round shift right by one limb is FREE: the accumulator limbs are
    referred to through a Python list that is rotated at generation time, so
    "t[j-1] = ..." becomes a renaming rather than S moves.
    """
    T = [f"t{j}" for j in range(S + 2)]
    L = [f"{ind}{t} = 0;" for t in T]
    for i in range(S):
        L.append(f"{ind}// --- CIOS round {i} ---")
        L += _mac_row(T, a, f"{b}{i}", ind)
        # m makes t + m*p divisible by 2^32, so the low word of that product
        # cancels t[0] exactly; T[0] is written anyway because the SUBTRACTION
        # it performs is what sets the carry the rest of the chain needs.
        L.append(f"{ind}m = {T[0]} * {NP};")
        L += _mac_row_const(T, P32, "m", ind)
        # Shift right one limb by rotating the names; the vacated top is zero.
        T = T[1:] + [T[0]]
        L.append(f"{ind}{T[S+1]} = 0;")
    L += _cond_sub(dst, T, ind)
    return L


def _mac_row_const(T, limbs, mult, ind):
    """`T += limbs * mult` where `limbs` is a list of compile-time constants."""
    L = [f"{ind}// t += m * p  (lo pass, then hi pass one limb up)"]
    L.append(f"{ind}{T[0]} = mad_lo_cc_u32({limbs[0]}, {mult}, {T[0]});")
    for j in range(1, S):
        L.append(f"{ind}{T[j]} = madc_lo_cc_u32({limbs[j]}, {mult}, {T[j]});")
    L.append(f"{ind}{T[S]} = addc_cc_u32({T[S]}, 0);")
    L.append(f"{ind}{T[S+1]} = addc_u32({T[S+1]}, 0);")
    L.append(f"{ind}{T[1]} = mad_hi_cc_u32({limbs[0]}, {mult}, {T[1]});")
    for j in range(1, S):
        L.append(f"{ind}{T[j+1]} = madc_hi_cc_u32({limbs[j]}, {mult}, {T[j+1]});")
    L.append(f"{ind}{T[S+1]} = addc_u32({T[S+1]}, 0);")
    return L


def add_mod_cc(dst, a, b, ind="    "):
    L = [f"{ind}t0 = add_cc_u32({a}0, {b}0);"]
    for j in range(1, S):
        L.append(f"{ind}t{j} = addc_cc_u32({a}{j}, {b}{j});")
    # a, b < p < 2^254, so a + b < 2^255 and limb 7 cannot carry out.
    L += _cond_sub(dst, [f"t{j}" for j in range(S)], ind)
    return L


def sub_mod_cc(dst, a, b, ind="    "):
    L = [f"{ind}t0 = sub_cc_u32({a}0, {b}0);"]
    for j in range(1, S):
        L.append(f"{ind}t{j} = subc_cc_u32({a}{j}, {b}{j});")
    L.append(f"{ind}mask = subc_u32(0, 0);")
    L.append(f"{ind}// mask is all-ones exactly when the subtraction went negative.")
    # Build every masked modulus limb BEFORE the add chain starts. They could
    # legally be built inside it -- `and` does not write CC -- but keeping the
    # chain a contiguous run of `.cc` instructions is the property that is easy
    # to check by eye and easy to preserve.
    for j in range(S):
        L.append(f"{ind}d{j} = {P32[j]} & mask;")
    L.append(f"{ind}{dst}0 = add_cc_u32(t0, d0);")
    for j in range(1, S):
        L.append(f"{ind}{dst}{j} = addc_cc_u32(t{j}, d{j});")
    return L


def mont_mul_u64(dst, a, b, ind="    "):
    """CIOS, fully unrolled. `a` and `b` name eight U32 vars each; `dst` names
    eight already-declared U32 vars to receive the result.

    Scratch (`t0..t9`, `c`, `m`, `pr`, `sw`) must already be declared by the
    caller, once per kernel, so the unrolling does not allocate 128 registers'
    worth of temporaries.
    """
    L = []
    e = L.append
    for j in range(S + 2):
        e(f"{ind}t{j} = 0;")
    for i in range(S):
        e(f"{ind}// --- CIOS round {i} ---")
        e(f"{ind}c = 0;")
        for j in range(S):
            # Maximum is (2^32-1)^2 + 2*(2^32-1) = 2^64 - 1 exactly, so this
            # cannot overflow a u64. That is the property CIOS is built on.
            e(f"{ind}pr = mul_wide_u32({a}{j}, {b}{i}) + t{j} + c;")
            e(f"{ind}t{j} = u64_lo32(pr);")
            e(f"{ind}c = u64_hi32(pr);")
        e(f"{ind}sw = t{S};")
        e(f"{ind}pr = sw + c;")
        e(f"{ind}t{S} = u64_lo32(pr);")
        e(f"{ind}t{S+1} = u64_hi32(pr);")
        # m makes t + m*p divisible by 2^32; storing to t[j-1] is the shift.
        e(f"{ind}m = t0 * {NP};")
        e(f"{ind}pr = mul_wide_u32(m, {P32[0]}) + t0;")
        e(f"{ind}c = u64_hi32(pr);")
        for j in range(1, S):
            e(f"{ind}pr = mul_wide_u32(m, {P32[j]}) + t{j} + c;")
            e(f"{ind}t{j-1} = u64_lo32(pr);")
            e(f"{ind}c = u64_hi32(pr);")
        e(f"{ind}sw = t{S};")
        e(f"{ind}pr = sw + c;")
        e(f"{ind}t{S-1} = u64_lo32(pr);")
        e(f"{ind}t{S} = t{S+1} + u64_hi32(pr);")
    # t < 2p and 2p < 2^255, so limb 8 is zero: one conditional subtract.
    e(f"{ind}// t is below 2p; subtract p once if it is at least p.")
    e(f"{ind}c = 0;")
    for j in range(S):
        e(f"{ind}sw = t{j};")
        e(f"{ind}pr = sw - {P32[j]} - c;")
        e(f"{ind}d{j} = u64_lo32(pr);")
        e(f"{ind}c = u64_hi32(pr) & 1;")
    # c == 0 means no borrow out, i.e. t >= p, so take the difference.
    e(f"{ind}mask = 0 - (1 - c);")
    e(f"{ind}nmask = 0 - c;")
    for j in range(S):
        e(f"{ind}{dst}{j} = (d{j} & mask) | (t{j} & nmask);")
    return L


def add_mod_u64(dst, a, b, ind="    "):
    """dst = (a + b) mod p, eight limbs, branch-free."""
    L = []
    e = L.append
    e(f"{ind}c = 0;")
    for j in range(S):
        e(f"{ind}sw = {a}{j};")
        e(f"{ind}pr = sw + {b}{j} + c;")
        e(f"{ind}t{j} = u64_lo32(pr);")
        e(f"{ind}c = u64_hi32(pr);")
    # a, b < p < 2^254, so a + b < 2^255 and the carry out of limb 7 is zero.
    e(f"{ind}c = 0;")
    for j in range(S):
        e(f"{ind}sw = t{j};")
        e(f"{ind}pr = sw - {P32[j]} - c;")
        e(f"{ind}d{j} = u64_lo32(pr);")
        e(f"{ind}c = u64_hi32(pr) & 1;")
    e(f"{ind}mask = 0 - (1 - c);")
    e(f"{ind}nmask = 0 - c;")
    for j in range(S):
        e(f"{ind}{dst}{j} = (d{j} & mask) | (t{j} & nmask);")
    return L


def sub_mod_u64(dst, a, b, ind="    "):
    """dst = (a - b) mod p. Add p back when the subtraction borrowed."""
    L = []
    e = L.append
    e(f"{ind}c = 0;")
    for j in range(S):
        e(f"{ind}sw = {a}{j};")
        e(f"{ind}pr = sw - {b}{j} - c;")
        e(f"{ind}t{j} = u64_lo32(pr);")
        e(f"{ind}c = u64_hi32(pr) & 1;")
    e(f"{ind}// c is the borrow out: 1 means the true result was negative.")
    e(f"{ind}mask = 0 - c;")
    e(f"{ind}nc = 0;")
    for j in range(S):
        e(f"{ind}sw = t{j};")
        e(f"{ind}pr = sw + ({P32[j]} & mask) + nc;")
        e(f"{ind}{dst}{j} = u64_lo32(pr);")
        e(f"{ind}nc = u64_hi32(pr);")
    return L


# The single switch. Every kernel below calls these, so `CARRY` re-generates
# the whole family either way and `tools/` stays the only place the choice is
# made.
def mont_mul(dst, a, b, ind="    "):
    return (mont_mul_cc if CARRY else mont_mul_u64)(dst, a, b, ind)


def add_mod(dst, a, b, ind="    "):
    return (add_mod_cc if CARRY else add_mod_u64)(dst, a, b, ind)


def sub_mod(dst, a, b, ind="    "):
    return (sub_mod_cc if CARRY else sub_mod_u64)(dst, a, b, ind)


def scratch_decls(ind="    "):
    L = [f"{ind}// Scratch, declared once so unrolling does not allocate a",
         f"{ind}// register per step."]
    if not CARRY:
        # `pr` and `sw` are the 64-bit lanes that make carries visible when
        # there are no carry-flag intrinsics: no operator exposes them.
        L += [f"{ind}let pr: U64 = 0;", f"{ind}let sw: U64 = 0;",
              f"{ind}let c: U32 = 0;", f"{ind}let nc: U32 = 0;"]
    L += [f"{ind}let m: U32 = 0;",
          f"{ind}let mask: U32 = 0;", f"{ind}let nmask: U32 = 0;"]
    for j in range(S + 2):
        L.append(f"{ind}let t{j}: U32 = 0;")
    for j in range(S):
        L.append(f"{ind}let d{j}: U32 = 0;")
    return L


# PLANAR ARRAYS OF uint4. Limb `j` of element `i` lives at
# `[(j//4) * 4*count + i*4 + (j%4)]` - i.e. two planes, each an array of
# 4-limb vectors.
#
# This layout is chosen by the profiler, not by taste, and it is the third one
# tried. NCU showed the kernel at 9.73% compute throughput with 45.1 of every
# 64.3 warp cycles stalled on the local/global instruction queue being full -
# too many separate memory INSTRUCTIONS, not too few bytes.
#
#   array-of-structs   [i*8 + j]      one warp instruction touches 32 sectors
#                                     32 bytes apart and uses 4 bytes of each.
#   struct-of-arrays   [j*count + i]  perfectly coalesced, but still EIGHT
#                                     instructions per element, so the queue
#                                     pressure barely moved at large N.
#   planar uint4       this one       two `ld.global.v4.u32` per element, and
#                                     a warp's 32 lanes are 512 CONTIGUOUS,
#                                     fully-used bytes per instruction.
#
# ptxas will not merge the scalar form itself: every `block_ptr2d_load` carries
# its own bounds predicate, and a predicated load is not a merge candidate.
# Hence the explicit v4 intrinsic.
def load(var, buf, idx, count, ind="    "):
    L = []
    for pl in range(S // 4):
        L.append(f"{ind}let {var}v{pl}: U32x4 = block_ptr2d_load_v4("
                 f"{buf}, {pl}, ({idx}) * 4, 4 * {count}, 2, 4 * {count});")
        for lane in range(4):
            L.append(f"{ind}let {var}{pl * 4 + lane}: U32 = "
                     f"{var}v{pl}.{'xyzw'[lane]};")
    return L


def store(var, buf, idx, count, ind="    "):
    return [f"{ind}block_ptr2d_store_v4({buf}, {pl}, ({idx}) * 4, 4 * {count}, "
            f"2, 4 * {count}, {var}{pl*4}, {var}{pl*4+1}, {var}{pl*4+2}, "
            f"{var}{pl*4+3});" for pl in range(S // 4)]


HEADER = """// GENERATED by tools/gen_bn254_kernels.py -- do not edit by hand.
//
// BN254 Fr arithmetic with every limb in a register and the modulus as
// immediates. The limbs are unrolled because this backend has no local arrays
// and `let` binds a name to one register, so a loop over limb indices cannot
// be expressed; `tests/bn254_fr_mul.ysu` is the readable rolled version, which
// keeps its accumulator in global scratch and is ~%dx slower for that reason
// alone.
//
// Checked against src/zk_field.rs on the device by tests/zk_gpu_field.rs.
"""


def gen_mul():
    L = [HEADER.replace("%d", "N"), "",
         "kernel bn254_fr_mul_fast(",
         "    A: GlobalMemory<U32>,",
         "    B: GlobalMemory<U32>,",
         "    Out: GlobalMemory<U32>,",
         "    N: I32",
         ") {",
         "    let tid: I32 = block_idx_x() * 256 + thread_idx_x();"]
    L += load("a", "A", "tid", "N")
    L += load("b", "B", "tid", "N")
    L += [f"    let r{j}: U32 = 0;" for j in range(S)]
    L += scratch_decls()
    L += mont_mul("r", "a", "b")
    L += store("r", "Out", "tid", "N")
    L += ["}", "", "fn main() {}", ""]
    return "\n".join(L)


def gen_ntt():
    """One radix-2 NTT stage. The host launches it once per stage.

    Each thread owns one butterfly: it reads the pair (i, i + half), multiplies
    the odd element by a twiddle, and writes back the sum and the difference.
    `Tw` holds the twiddle for this stage's block, already in Montgomery form
    so that the Montgomery multiply returns a canonical value.
    """
    L = [HEADER.replace("%d", "N"), "",
         "// One radix-2 decimation-in-time butterfly per thread.",
         "//   u = x[i]                v = x[i + half] * w",
         "//   x[i] = u + v            x[i + half] = u - v",
         "// `half` and `Log2Half` describe the current stage; `Tw` is indexed",
         "// by the butterfly's position within its block.",
         "kernel bn254_ntt_stage(",
         "    X: GlobalMemory<U32>,",
         "    Tw: GlobalMemory<U32>,",
         "    Half: I32,",
         "    NPairs: I32,",
         "    NElem: I32",
         ") {",
         "    let gid: I32 = block_idx_x() * 256 + thread_idx_x();",
         "    // Split the flat butterfly id into (block, position in block).",
         "    let blk: I32 = gid / Half;",
         "    let pos: I32 = gid % Half;",
         "    let i0: I32 = blk * Half * 2 + pos;",
         "    let i1: I32 = i0 + Half;"]
    L += load("u", "X", "i0", "NElem")
    L += load("q", "X", "i1", "NElem")
    L += load("w", "Tw", "pos", "Half")
    L += [f"    let v{j}: U32 = 0;" for j in range(S)]
    L += [f"    let s{j}: U32 = 0;" for j in range(S)]
    L += [f"    let e{j}: U32 = 0;" for j in range(S)]
    L += scratch_decls()
    L += ["    // v = x[i1] * w  (Montgomery: w carries the R factor)"]
    L += mont_mul("v", "q", "w")
    L += ["    // s = u + v,  e = u - v"]
    L += add_mod("s", "u", "v")
    L += sub_mod("e", "u", "v")
    L += store("s", "X", "i0", "NElem")
    L += store("e", "X", "i1", "NElem")
    L += ["}", "", "fn main() {}", ""]
    return "\n".join(L)


def gen_sub_vec():
    """Out[i] = (A[i] - B[i]) mod p.

    The one pointwise operation the QAP witness map needs that is not already
    a multiply. Everything else folds into a table: the ifft's 1/n scaling and
    the coset offset g^i become one vector, and the vanishing-polynomial
    constant folds into the final one, so no separate scale kernel exists.
    """
    use_field(FR)
    L = [HEADER.replace("%d", "N"), "",
         "kernel bn254_sub_vec(",
         "    A: GlobalMemory<U32>,",
         "    B: GlobalMemory<U32>,",
         "    Out: GlobalMemory<U32>,",
         "    N: I32",
         ") {",
         "    let tid: I32 = block_idx_x() * 256 + thread_idx_x();"]
    L += load("a", "A", "tid", "N")
    L += load("b", "B", "tid", "N")
    L += [f"    let r{j}: U32 = 0;" for j in range(S)]
    L += scratch_decls()
    L += sub_mod("r", "a", "b")
    L += store("r", "Out", "tid", "N")
    L += ["}", "", "fn main() {}", ""]
    return "\n".join(L)


def gen_permute():
    """Gather `Dst[i] = Src[Perm[i]]`, 8 limbs at a time.

    Exists so that a CHAIN of transforms can stay on the device. The radix-2
    stage is decimation-in-time: it wants bit-reversed input and produces
    natural order. A single transform can be permuted on the host for free
    (the data is being uploaded anyway), but the Groth16 QAP witness map runs
    SEVEN transforms back to back, and round-tripping 2^21 elements over PCIe
    between each one costs more than the transforms save.

    The permutation is a table rather than bit-twiddling in the kernel: it is
    identical for all seven transforms, so it is built once on the host and
    uploaded once, and a table also serves the base-4 digit reversal the
    radix-4 stage needs without a second kernel.
    """
    use_field(FR)
    L = [HEADER.replace("%d", "N"), "",
         "// Dst[i] = Src[Perm[i]] over 8-limb elements in the planar layout.",
         "kernel bn254_permute(",
         "    Src: GlobalMemory<U32>,",
         "    Dst: GlobalMemory<U32>,",
         "    Perm: GlobalMemory<U32>,",
         "    NElem: I32",
         ") {",
         "    let i: I32 = block_idx_x() * 256 + thread_idx_x();",
         "    let j: U32 = block_ptr2d_load(Perm, 0, i, NElem, 1, NElem);"]
    L += load("v", "Src", "j", "NElem")
    L += store("v", "Dst", "i", "NElem")
    L += ["}", "", "fn main() {}", ""]
    return "\n".join(L)


def gen_ntt4():
    """One radix-4 Cooley-Tukey stage: one 4-point butterfly per thread.

    Halves the number of passes over memory against the radix-2 stage, which
    is the only thing that helps once a transform is DRAM-bound - at N = 2^23
    the radix-2 version sits at 82% of peak bandwidth, where no layout change
    moves fewer bytes. It also does LESS arithmetic: three twiddle multiplies
    per four points, where two radix-2 stages cost four.

    With `i` the primitive 4th root (omega_m^(m/4)) and the twiddled inputs
    A, B, C, D, the 4-point DFT is

        t0 = A + C     t2 = B + D
        t1 = A - C     t3 = (B - D) * i
        X0 = t0 + t2   X1 = t1 + t3   X2 = t0 - t2   X3 = t1 - t3

    Requires log2(N) EVEN, i.e. N = 4^k, and the input in base-4
    digit-reversed order. The host enforces both; an odd log2(N) needs a
    mixed-radix permutation that is not worth the risk of getting subtly
    wrong - it would still pass a round trip.
    """
    L = [HEADER.replace("%d", "N"), "",
         "kernel bn254_ntt4_stage(",
         "    X: GlobalMemory<U32>,",
         "    Tw1: GlobalMemory<U32>,",
         "    Tw2: GlobalMemory<U32>,",
         "    Tw3: GlobalMemory<U32>,",
         "    Iw: GlobalMemory<U32>,",
         "    Quarter: I32,",
         "    NElem: I32",
         ") {",
         "    let gid: I32 = block_idx_x() * 256 + thread_idx_x();",
         "    let blk: I32 = gid / Quarter;",
         "    let pos: I32 = gid % Quarter;",
         "    let i0: I32 = blk * Quarter * 4 + pos;",
         "    let i1: I32 = i0 + Quarter;",
         "    let i2: I32 = i1 + Quarter;",
         "    let i3: I32 = i2 + Quarter;"]
    L += [f"    let A{j}: U32 = 0;" for j in range(S)]
    L += [f"    let B{j}: U32 = 0;" for j in range(S)]
    L += [f"    let C{j}: U32 = 0;" for j in range(S)]
    L += [f"    let D{j}: U32 = 0;" for j in range(S)]
    L += [f"    let E{j}: U32 = 0;" for j in range(S)]   # t0 / X0
    L += [f"    let F{j}: U32 = 0;" for j in range(S)]   # t1 / X1
    L += [f"    let G{j}: U32 = 0;" for j in range(S)]   # t2 / X2
    L += [f"    let H{j}: U32 = 0;" for j in range(S)]   # t3 / X3
    L += scratch_decls()

    # A = x[i0]; the other three are twiddled as they are read, so each
    # twiddle dies immediately and never competes for a register.
    L += load("ld", "X", "i0", "NElem")
    L += [f"    A{j} = ld{j};" for j in range(S)]
    L += load("qb", "X", "i1", "NElem")
    L += load("wb", "Tw1", "pos", "Quarter")
    L += mont_mul("B", "qb", "wb")
    L += load("qc", "X", "i2", "NElem")
    L += load("wc", "Tw2", "pos", "Quarter")
    L += mont_mul("C", "qc", "wc")
    L += load("qd", "X", "i3", "NElem")
    L += load("wd", "Tw3", "pos", "Quarter")
    L += mont_mul("D", "qd", "wd")

    L += ["    // t0 = A + C, t1 = A - C, t2 = B + D, t3 = (B - D) * i"]
    L += add_mod("E", "A", "C")
    L += sub_mod("F", "A", "C")
    L += add_mod("G", "B", "D")
    L += sub_mod("A", "B", "D")           # A is dead; reuse it for (B - D)
    L += load("iw", "Iw", "0", "1")
    L += mont_mul("H", "A", "iw")

    L += ["    // X0 = t0 + t2, X1 = t1 + t3, X2 = t0 - t2, X3 = t1 - t3"]
    L += add_mod("B", "E", "G")
    L += add_mod("C", "F", "H")
    L += sub_mod("D", "E", "G")
    L += sub_mod("E", "F", "H")
    L += store("B", "X", "i0", "NElem")
    L += store("C", "X", "i1", "NElem")
    L += store("D", "X", "i2", "NElem")
    L += store("E", "X", "i3", "NElem")
    L += ["}", "", "fn main() {}", ""]
    return "\n".join(L)


# ── Shared-memory stage fusion ────────────────────────────────────────────
#
# NCU says `bn254_ntt4_stage` is DRAM-bound at 91% of peak, so there is
# nothing left to win INSIDE it; the only lever left is running fewer passes
# over the array. At N = 2^22 radix-4 needs 11 stages and therefore 11
# read+write passes, 2.95 GB. icicle does the same transform in what works out
# to ~6.2 equivalent passes, and the difference is entirely that it keeps
# intermediates in shared memory across several butterfly stages.
#
# A CTA that holds FUSED_ELEMS contiguous elements can run every stage whose
# quarter-stride divides that span, because such a stage's butterflies never
# reach outside it. FUSED_ELEMS = 4^5 = 1024 elements is 32 KB (each element is
# eight u32), just under the 48 KB static per-CTA limit, and covers five
# stages: q = 1, 4, 16, 64, 256.
FUSED_LOG = 5                       # radix-4 stages fused into one launch
FUSED_ELEMS = 4 ** FUSED_LOG        # 1024 elements per CTA
FUSED_THREADS = FUSED_ELEMS // 4    # 256 threads, one 4-point butterfly each
# Twiddle arrays for the fused stages are CONCATENATED, stage t occupying
# 4^t entries at this offset. One array rather than five parameters.
TW_OFF = [(4 ** t - 1) // 3 for t in range(FUSED_LOG)]
TW_TOTAL = (4 ** FUSED_LOG - 1) // 3


def swizzle(dst, src, ind="    "):
    """`dst = src ^ ((src >> 2) & 7)`, the shared-memory slot permutation.

    Without it the low stages conflict badly. Shared memory is 32 banks of 4
    bytes, so a 128-bit access is resolved in groups of 8 threads and those 8
    must land on 8 distinct 16-byte slots (mod 8) to be conflict-free. At
    q = 1 a thread owns elements 4*tid..4*tid+3, so slot mod 8 is {0, 4} for
    the whole group -- a 4-way conflict on the very stage that runs first.

    XOR-ing in bits 2..4 fixes every fused stage at once. Checked by hand for
    q = 1, 4, 16, 64, 256: each gives 8 distinct residues per group. It is a
    bijection on [0, 1024) because bits 3 and above pass through unchanged, so
    the map can be inverted bit by bit -- which is what makes it safe to apply
    on store and load without tracking where anything went.
    """
    return [f"{ind}let {dst}: I32 = {src} ^ (({src} >> 2) & 7);"]


def sload(var, base, slot, ind="    "):
    """Read one 8-limb element from shared memory at swizzled slot `slot`.

    Planar, exactly like the global layout: plane 0 holds limbs 0-3 of every
    element in the first FUSED_ELEMS 16-byte slots, plane 1 holds limbs 4-7 in
    the next FUSED_ELEMS. Two `ld.shared.v4.u32` per element.
    """
    L = []
    for pl in range(S // 4):
        L.append(f"{ind}let {var}v{pl}: U32x4 = shared_load_v4("
                 f"{base}, {slot} + {pl * FUSED_ELEMS});")
        for lane in range(4):
            L.append(f"{ind}let {var}{pl * 4 + lane}: U32 = "
                     f"{var}v{pl}.{'xyzw'[lane]};")
    return L


def sstore(var, base, slot, ind="    "):
    return [f"{ind}shared_store_v4({base}, {slot} + {pl * FUSED_ELEMS}, "
            f"{var}{pl*4}, {var}{pl*4+1}, {var}{pl*4+2}, {var}{pl*4+3});"
            for pl in range(S // 4)]


def gen_ntt4_fused():
    """FUSED_LOG radix-4 stages in one launch, with shared memory between them.

    Same butterfly as `gen_ntt4`, same twiddle convention, same digit-reversed
    input. The only difference is where the intermediates live: CTA b owns
    global elements [b*FUSED_ELEMS, (b+1)*FUSED_ELEMS), stages them into shared
    memory once, runs five stages there, and writes back once. One pass over
    DRAM instead of five.

    Why one barrier per stage and not two: at every fused stage a thread's four
    slots are exactly the four it will overwrite, and those quadruples
    partition the CTA's 1024 slots. So no thread ever reads a slot another
    thread is writing WITHIN a stage, and the only ordering that needs
    enforcing is stage t's writes against stage t+1's reads.

    The last stage skips shared memory on the way out. At q = 256 a thread owns
    slots {tid, tid+256, tid+512, tid+768}, which is precisely the coalesced
    global pattern the staging loop used on the way in -- so it can store
    straight to X. The first stage cannot do the mirror of this: at q = 1 a
    thread owns four CONSECUTIVE elements, and reading those from global would
    have a warp touching 64-byte-strided addresses in each of its four load
    instructions.

    Twiddle index. In the unfused kernel it is `i0 mod q`. Here i0 is
    `b*FUSED_ELEMS + l0`, and FUSED_ELEMS = 4^FUSED_LOG is divisible by every
    fused q, so `i0 mod q == l0 mod q` and the local index is enough.
    """
    L = [HEADER.replace("%d", "N"), "",
         f"// {FUSED_LOG} radix-4 stages fused through shared memory:",
         f"// {FUSED_ELEMS} elements/CTA * {S} u32 = "
         f"{FUSED_ELEMS * S * 4 // 1024} KB, {FUSED_THREADS} threads.",
         "//",
         "// Tw1/Tw2/Tw3 are the per-stage twiddle tables CONCATENATED, stage t",
         f"// at offset {TW_OFF} with 4^t entries ({TW_TOTAL} total).",
         "kernel bn254_ntt4_fused(",
         "    X: GlobalMemory<U32>,",
         "    Tw1: GlobalMemory<U32>,",
         "    Tw2: GlobalMemory<U32>,",
         "    Tw3: GlobalMemory<U32>,",
         "    Iw: GlobalMemory<U32>,",
         "    NElem: I32",
         ") {",
         f"    let smem: U64 = shared_alloc_u32({FUSED_ELEMS * S});",
         "    let tid: I32 = thread_idx_x();",
         f"    let chunk: I32 = block_idx_x() * {FUSED_ELEMS};"]
    for v in "ABCDEFGH":
        L += [f"    let {v}{j}: U32 = 0;" for j in range(S)]
    L += scratch_decls()

    # ── stage the CTA's slice into shared memory, coalesced ──
    L += ["    // Coalesced staging: thread `tid` takes elements",
          f"    // tid, tid+{FUSED_THREADS}, ... so a warp reads contiguous bytes."]
    for e in range(FUSED_ELEMS // FUSED_THREADS):
        L += [f"    let li{e}: I32 = tid + {e * FUSED_THREADS};"]
        L += swizzle(f"ls{e}", f"li{e}")
        L += load(f"g{e}", "X", f"chunk + li{e}", "NElem")
        L += sstore(f"g{e}", "smem", f"ls{e}")
    L += ["    barrier_sync();"]

    for t in range(FUSED_LOG):
        q = 4 ** t
        last = t == FUSED_LOG - 1
        L += ["", f"    // ── fused stage {t}: quarter = {q} ──"]
        L += [f"    let pos{t}: I32 = tid & {q - 1};",
              f"    let l0_{t}: I32 = ((tid >> {2 * t}) * {4 * q}) + pos{t};",
              f"    let l1_{t}: I32 = l0_{t} + {q};",
              f"    let l2_{t}: I32 = l1_{t} + {q};",
              f"    let l3_{t}: I32 = l2_{t} + {q};"]
        for k in range(4):
            L += swizzle(f"s{k}_{t}", f"l{k}_{t}")

        L += sload("ld", "smem", f"s0_{t}")
        L += [f"    A{j} = ld{j};" for j in range(S)]
        if q == 1:
            # Stage 0 has ONE twiddle position, and w^0 = 1. Multiplying by
            # Montgomery one is the identity, so three of this stage's four
            # CIOS multiplies are computing `x * R * R^-1`. Dropping them is
            # exact, not an approximation.
            #
            # Worth naming because it is 3 of the 20 multiplies in the whole
            # fused kernel, and NCU says that kernel is SM-bound at 73% -- so
            # it is 15% of the arithmetic on the critical path, where in the
            # unfused kernel (DRAM-bound at 91%) the same saving would have
            # bought nothing at all. The optimisation became worth doing
            # because the bottleneck moved.
            L += ["    // w^0 = 1: the three twiddle multiplies here are the identity."]
            L += sload("qb", "smem", f"s1_{t}")
            L += [f"    B{j} = qb{j};" for j in range(S)]
            L += sload("qc", "smem", f"s2_{t}")
            L += [f"    C{j} = qc{j};" for j in range(S)]
            L += sload("qd", "smem", f"s3_{t}")
            L += [f"    D{j} = qd{j};" for j in range(S)]
        else:
            L += sload("qb", "smem", f"s1_{t}")
            L += load("wb", "Tw1", f"{TW_OFF[t]} + pos{t}", str(TW_TOTAL))
            L += mont_mul("B", "qb", "wb")
            L += sload("qc", "smem", f"s2_{t}")
            L += load("wc", "Tw2", f"{TW_OFF[t]} + pos{t}", str(TW_TOTAL))
            L += mont_mul("C", "qc", "wc")
            L += sload("qd", "smem", f"s3_{t}")
            L += load("wd", "Tw3", f"{TW_OFF[t]} + pos{t}", str(TW_TOTAL))
            L += mont_mul("D", "qd", "wd")

        L += ["    // t0 = A + C, t1 = A - C, t2 = B + D, t3 = (B - D) * i"]
        L += add_mod("E", "A", "C")
        L += sub_mod("F", "A", "C")
        L += add_mod("G", "B", "D")
        L += sub_mod("A", "B", "D")
        L += load("iw", "Iw", "0", "1")
        L += mont_mul("H", "A", "iw")

        L += ["    // X0 = t0 + t2, X1 = t1 + t3, X2 = t0 - t2, X3 = t1 - t3"]
        L += add_mod("B", "E", "G")
        L += add_mod("C", "F", "H")
        L += sub_mod("D", "E", "G")
        L += sub_mod("E", "F", "H")

        if last:
            # q == FUSED_THREADS here, so l0..l3 are tid, tid+256, tid+512,
            # tid+768 -- the coalesced pattern. Straight out to global.
            for k, v in enumerate("BCDE"):
                L += store(v, "X", f"chunk + l{k}_{t}", "NElem")
        else:
            for k, v in enumerate("BCDE"):
                L += sstore(v, "smem", f"s{k}_{t}")
            L += ["    barrier_sync();"]

    L += ["}", "", "fn main() {}", ""]
    return "\n".join(L)


# ── Strided stage fusion, for the stages the contiguous one cannot reach ──
#
# After `gen_ntt4_fused` there are still `log4(N) - FUSED_LOG` stages left, and
# at N = 2^22 they are 73% of the remaining time. Their quarter-stride exceeds
# a CTA's 1024-element slice, so a CTA of CONTIGUOUS elements cannot hold a
# whole butterfly.
#
# A CTA of STRIDED elements can. With `Q0` the smallest quarter in the group,
# take the 1024 elements
#
#     sb*Q0*256 + p0 + u + j*Q0     for u in [0, 4), j in [0, 256)
#
# In j-space a stage of quarter `Q0 * 4^t` is an ordinary radix-4 butterfly of
# quarter `4^t`, so t = 0..3 fuses HIGH_LOG = 4 of them.
#
# The four `u` are what keeps it coalesced. Four consecutive threads take
# u = 0,1,2,3 at the same j, so they read four CONSECUTIVE elements - 64
# contiguous bytes per plane, fully-used sectors. Without them a warp would
# ask for 32 separate 16-byte pieces 16 KB apart and use half of every sector
# the memory system fetched.
#
# `Q0` is a RUNTIME parameter, and that is a testability decision, not a
# generality one. This kernel only applies when four stages sit above the fused
# low five, i.e. N >= 4^9, and a naive DFT at 2^18 is ~7e10 field multiplies.
# With Q0 passed in, the identical kernel runs at Q0 = 4 and N = 4096, where
# the DFT oracle costs 16.7M multiplies and every line of the index derivation
# is exercised. Checking it only against Y's own unfused path would share
# exactly the class of bug the DFT exists to catch.
# The kernel is generated at several DEPTHS, because after the low five, stages
# come in groups of HIGH_MAX and `log4(N) - FUSED_LOG` is not a multiple of it.
# A depth-h variant holds the same 1024 elements as JV = 4^h j-values across
# U = 1024/JV consecutive base values, so the CTA size, the thread count and
# the shared-memory footprint are identical whatever h is -- only the shape of
# the slice changes. Emitting depths 2..HIGH_MAX means every remainder can be
# fused, which is what takes 11 passes to 3 rather than to 4.
#
# Depth 1 is deliberately NOT emitted: fusing one stage is the unfused kernel
# plus a shared-memory round trip, i.e. strictly worse. A one-stage remainder
# runs `bn254_ntt4_stage`.
HIGH_MAX = 4
HIGH_DEPTHS = list(range(2, HIGH_MAX + 1))
HIGH_ELEMS = FUSED_ELEMS            # 1024 elements -- same 32 KB as the low one
HIGH_THREADS = HIGH_ELEMS // 4      # 256, whatever the depth


def high_geom(h):
    """(j-values, consecutive base values) for a depth-`h` strided kernel."""
    jv = 4 ** h
    return jv, HIGH_ELEMS // jv


def twh_off(h):
    """Stage t's offset into the concatenated table, as a multiple of Q0."""
    return [(4 ** t - 1) // 3 for t in range(h)]


def twh_total(h):
    return (4 ** h - 1) // 3


def gen_ntt4_fused_high(h):
    """`h` radix-4 stages of quarter Q0*4^t, fused through shared memory.

    Same butterfly, same twiddle convention and the same shared-memory layout
    as `gen_ntt4_fused`; only the map from thread to element differs.

    Local slot is `j*U + u`, which mirrors the global order within the CTA and
    keeps the staging access pattern identical to the low kernel's. The same
    XOR swizzle applies. It is not quite as clean here - at t = 1 the eight
    threads of an access group land on six distinct residues rather than eight,
    a 2-way conflict on one of four stages - but the depth-4 kernel measures
    1.8% conflicted wavefronts overall against the low kernel's 0.95%, so the
    headroom is there.

    The last stage writes straight to global for the same reason as the low
    kernel: at t = h-1 a thread owns `j = bfly + k*4^(h-1)` at its own `u`,
    which is exactly the coalesced pattern the staging loop read in.
    """
    JV, U = high_geom(h)
    OFF, TOT = twh_off(h), twh_total(h)
    name = f"bn254_ntt4_fused_high{h}"
    L = [HEADER.replace("%d", "N"), "",
         f"// {h} STRIDED radix-4 stages fused through shared memory:",
         f"// quarters Q0*4^t for t in 0..{h}, {HIGH_ELEMS} elements/CTA",
         f"// ({HIGH_ELEMS * S * 4 // 1024} KB), {HIGH_THREADS} threads.",
         "//",
         f"// A CTA owns  sb*Q0*{JV} + p0 + u + j*Q0  for u in [0,{U}), j in [0,{JV}).",
         "// Tw1/Tw2/Tw3 are the per-stage tables CONCATENATED, stage t at",
         f"// offset Q0*{OFF} with Q0*4^t entries (Q0*{TOT} total).",
         f"kernel {name}(",
         "    X: GlobalMemory<U32>,",
         "    Tw1: GlobalMemory<U32>,",
         "    Tw2: GlobalMemory<U32>,",
         "    Tw3: GlobalMemory<U32>,",
         "    Iw: GlobalMemory<U32>,",
         "    Q0: I32,",
         "    NElem: I32",
         ") {",
         f"    let smem: U64 = shared_alloc_u32({HIGH_ELEMS * S});",
         "    let tid: I32 = thread_idx_x();",
         f"    let twlen: I32 = Q0 * {TOT};",
         f"    let pgn: I32 = Q0 / {U};",
         "    let cta: I32 = block_idx_x();",
         "    let pg: I32 = cta % pgn;",
         "    let sb: I32 = cta / pgn;",
         f"    let p0: I32 = pg * {U};",
         f"    let span: I32 = Q0 * {JV};",
         "    let sbase: I32 = sb * span + p0;",
         f"    let u: I32 = tid & {U - 1};",
         "    let pu: I32 = p0 + u;",
         f"    let bfly: I32 = tid / {U};"]
    for v in "ABCDEFGH":
        L += [f"    let {v}{j}: U32 = 0;" for j in range(S)]
    L += scratch_decls()

    L += [f"    // {U} consecutive threads share a j and take u = 0..{U-1}, so they",
          f"    // read {U} CONSECUTIVE elements: {U*16} contiguous bytes per plane."]
    per_thread = HIGH_ELEMS // HIGH_THREADS
    jstep = JV // per_thread
    for e in range(per_thread):
        L += [f"    let hj{e}: I32 = bfly + {e * jstep};",
              f"    let hl{e}: I32 = hj{e} * {U} + u;"]
        L += swizzle(f"hs{e}", f"hl{e}")
        L += load(f"g{e}", "X", f"sbase + u + hj{e} * Q0", "NElem")
        L += sstore(f"g{e}", "smem", f"hs{e}")
    L += ["    barrier_sync();"]

    for t in range(h):
        qj = 4 ** t
        last = t == h - 1
        L += ["", f"    // -- strided stage {t}: quarter = Q0 * {qj} --"]
        L += [f"    let jpos{t}: I32 = bfly & {qj - 1};",
              f"    let j0_{t}: I32 = ((bfly >> {2 * t}) * {4 * qj}) + jpos{t};",
              f"    let tw{t}: I32 = pu + Q0 * jpos{t};"]
        for k in range(4):
            L += [f"    let l{k}_{t}: I32 = (j0_{t} + {k * qj}) * {U} + u;"]
            L += swizzle(f"s{k}_{t}", f"l{k}_{t}")

        L += sload("ld", "smem", f"s0_{t}")
        L += [f"    A{j} = ld{j};" for j in range(S)]
        L += sload("qb", "smem", f"s1_{t}")
        L += load("wb", "Tw1", f"{OFF[t]} * Q0 + tw{t}", "twlen")
        L += mont_mul("B", "qb", "wb")
        L += sload("qc", "smem", f"s2_{t}")
        L += load("wc", "Tw2", f"{OFF[t]} * Q0 + tw{t}", "twlen")
        L += mont_mul("C", "qc", "wc")
        L += sload("qd", "smem", f"s3_{t}")
        L += load("wd", "Tw3", f"{OFF[t]} * Q0 + tw{t}", "twlen")
        L += mont_mul("D", "qd", "wd")

        L += ["    // t0 = A + C, t1 = A - C, t2 = B + D, t3 = (B - D) * i"]
        L += add_mod("E", "A", "C")
        L += sub_mod("F", "A", "C")
        L += add_mod("G", "B", "D")
        L += sub_mod("A", "B", "D")
        L += load("iw", "Iw", "0", "1")
        L += mont_mul("H", "A", "iw")

        L += ["    // X0 = t0 + t2, X1 = t1 + t3, X2 = t0 - t2, X3 = t1 - t3"]
        L += add_mod("B", "E", "G")
        L += add_mod("C", "F", "H")
        L += sub_mod("D", "E", "G")
        L += sub_mod("E", "F", "H")

        if last:
            for k, v in enumerate("BCDE"):
                L += [f"    let gj{k}_{t}: I32 = j0_{t} + {k * qj};"]
                L += store(v, "X", f"sbase + u + gj{k}_{t} * Q0", "NElem")
        else:
            for k, v in enumerate("BCDE"):
                L += sstore(v, "smem", f"s{k}_{t}")
            L += ["    barrier_sync();"]

    L += ["}", "", "fn main() {}", ""]
    return "\n".join(L)


def dbl_mod(dst, a, ind="    "):
    """dst = 2a mod p."""
    return add_mod(dst, a, a, ind)


def gen_g1_add():
    """BN254 G1 point addition in Jacobian coordinates, over Fq.

    `add-2007-bl` from the EFD: 11 multiplies and 5 squarings. The statement
    order below is chosen for LIVENESS, not readability - `Z3` is computed
    early so that Z1, Z2, Z1Z1 and Z2Z2 can all die before the second half,
    which keeps peak register pressure near the radix-4 kernel's rather than
    at the 11 simultaneous 8-limb temporaries the naive order needs.

    This is the atom of MSM. It is NOT complete addition: it assumes both
    inputs are non-zero and P != +-Q, which is the standard precondition
    Pippenger's bucket accumulation is arranged to satisfy. The test says so
    and only feeds it independent random points.
    """
    use_field(FQ)
    names = ["X1", "Y1", "Z1", "X2", "Y2", "Z2"]
    L = [HEADER.replace("%d", "N"), "",
         "// BN254 G1 Jacobian point addition over the BASE field Fq.",
         "// add-2007-bl; assumes P and Q are non-zero and P != +-Q.",
         "kernel bn254_g1_add(",
         "    PX: GlobalMemory<U32>,",
         "    PY: GlobalMemory<U32>,",
         "    PZ: GlobalMemory<U32>,",
         "    QX: GlobalMemory<U32>,",
         "    QY: GlobalMemory<U32>,",
         "    QZ: GlobalMemory<U32>,",
         "    RX: GlobalMemory<U32>,",
         "    RY: GlobalMemory<U32>,",
         "    RZ: GlobalMemory<U32>,",
         "    N: I32",
         ") {",
         "    let tid: I32 = block_idx_x() * 256 + thread_idx_x();"]
    for v, buf in zip(names, ["PX", "PY", "PZ", "QX", "QY", "QZ"]):
        L += load(v, buf, "tid", "N")
    # working temporaries
    temps = ["ZA", "ZB", "UA", "UB", "S1", "S2", "HH", "II", "JJ", "RR", "VV", "T1", "T2", "X3", "Y3", "Z3"]
    check_names(temps + names)
    for v in temps:
        L += [f"    let {v}{j}: U32 = 0;" for j in range(S)]
    L += scratch_decls()

    L += ["    // Z1Z1 = Z1^2 ; Z2Z2 = Z2^2"]
    L += mont_mul("ZA", "Z1", "Z1")
    L += mont_mul("ZB", "Z2", "Z2")
    L += ["    // U1 = X1*Z2Z2 ; U2 = X2*Z1Z1"]
    L += mont_mul("UA", "X1", "ZB")
    L += mont_mul("UB", "X2", "ZA")
    L += ["    // S1 = Y1*Z2*Z2Z2 ; S2 = Y2*Z1*Z1Z1"]
    L += mont_mul("T1", "Y1", "Z2")
    L += mont_mul("S1", "T1", "ZB")
    L += mont_mul("T1", "Y2", "Z1")
    L += mont_mul("S2", "T1", "ZA")
    L += ["    // H = U2 - U1"]
    L += sub_mod("HH", "UB", "UA")
    L += ["    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2) * H, computed here so that",
          "    // Z1, Z2, Z1Z1 and Z2Z2 can all die before the second half."]
    L += add_mod("T1", "Z1", "Z2")
    L += mont_mul("T2", "T1", "T1")
    L += sub_mod("T1", "T2", "ZA")
    L += sub_mod("T2", "T1", "ZB")
    L += mont_mul("Z3", "T2", "HH")
    L += ["    // I = (2H)^2 ; J = H*I ; r = 2*(S2-S1) ; V = U1*I"]
    L += dbl_mod("T1", "HH")
    L += mont_mul("II", "T1", "T1")
    L += mont_mul("JJ", "HH", "II")
    L += sub_mod("T1", "S2", "S1")
    L += dbl_mod("RR", "T1")
    L += mont_mul("VV", "UA", "II")
    L += ["    // X3 = r^2 - J - 2V"]
    L += mont_mul("T1", "RR", "RR")
    L += sub_mod("T2", "T1", "JJ")
    L += dbl_mod("T1", "VV")
    L += sub_mod("X3", "T2", "T1")
    L += ["    // Y3 = r*(V - X3) - 2*S1*J"]
    L += sub_mod("T1", "VV", "X3")
    L += mont_mul("T2", "RR", "T1")
    L += mont_mul("T1", "S1", "JJ")
    L += dbl_mod("UA", "T1")
    L += sub_mod("Y3", "T2", "UA")
    L += store("X3", "RX", "tid", "N")
    L += store("Y3", "RY", "tid", "N")
    L += store("Z3", "RZ", "tid", "N")
    L += ["}", "", "fn main() {}", ""]
    use_field(FR)
    return "\n".join(L)


def gen_g1_dbl():
    """BN254 G1 point doubling, `dbl-2009-l` (valid because BN254 has a = 0).

    Doubling needs its own formula: `add-2007-bl` computes H = U2 - U1, which
    is zero when P == Q, and then divides the geometry by it. Feeding a
    doubling to the add kernel does not produce a wrong point, it produces the
    point at infinity - silently. Pippenger needs both.

        A = X1^2   B = Y1^2   C = B^2
        D = 2*((X1+B)^2 - A - C)      E = 3*A       F = E^2
        X3 = F - 2D    Y3 = E*(D - X3) - 8C    Z3 = 2*Y1*Z1
    """
    use_field(FQ)
    L = [HEADER.replace("%d", "N"), "",
         "// BN254 G1 Jacobian point doubling over the base field Fq (a = 0).",
         "kernel bn254_g1_dbl(",
         "    PX: GlobalMemory<U32>,",
         "    PY: GlobalMemory<U32>,",
         "    PZ: GlobalMemory<U32>,",
         "    RX: GlobalMemory<U32>,",
         "    RY: GlobalMemory<U32>,",
         "    RZ: GlobalMemory<U32>,",
         "    N: I32",
         ") {",
         "    let tid: I32 = block_idx_x() * 256 + thread_idx_x();"]
    L += load("X1", "PX", "tid", "N")
    L += load("Y1", "PY", "tid", "N")
    L += load("Z1", "PZ", "tid", "N")
    temps = ["AA", "BB", "CC", "DD", "EE", "FF", "T1", "T2", "X3", "Y3", "Z3"]
    check_names(temps + ["X1", "Y1", "Z1"])
    for v in temps:
        L += [f"    let {v}{j}: U32 = 0;" for j in range(S)]
    L += scratch_decls()
    L += ["    // A = X1^2 ; B = Y1^2 ; C = B^2"]
    L += mont_mul("AA", "X1", "X1")
    L += mont_mul("BB", "Y1", "Y1")
    L += mont_mul("CC", "BB", "BB")
    L += ["    // D = 2*((X1+B)^2 - A - C)"]
    L += add_mod("T1", "X1", "BB")
    L += mont_mul("T2", "T1", "T1")
    L += sub_mod("T1", "T2", "AA")
    L += sub_mod("T2", "T1", "CC")
    L += dbl_mod("DD", "T2")
    L += ["    // E = 3A ; F = E^2"]
    L += dbl_mod("T1", "AA")
    L += add_mod("EE", "T1", "AA")
    L += mont_mul("FF", "EE", "EE")
    L += ["    // X3 = F - 2D"]
    L += dbl_mod("T1", "DD")
    L += sub_mod("X3", "FF", "T1")
    L += ["    // Y3 = E*(D - X3) - 8C"]
    L += sub_mod("T1", "DD", "X3")
    L += mont_mul("T2", "EE", "T1")
    L += dbl_mod("T1", "CC")
    L += dbl_mod("CC", "T1")
    L += dbl_mod("T1", "CC")
    L += sub_mod("Y3", "T2", "T1")
    L += ["    // Z3 = 2*Y1*Z1"]
    L += mont_mul("T1", "Y1", "Z1")
    L += dbl_mod("Z3", "T1")
    L += store("X3", "RX", "tid", "N")
    L += store("Y3", "RY", "tid", "N")
    L += store("Z3", "RZ", "tid", "N")
    L += ["}", "", "fn main() {}", ""]
    use_field(FR)
    return "\n".join(L)


def g1_add_body(dst, a, b, ind="    "):
    """`add-2007-bl` writing to dst{X,Y,Z}, reading a{...} and b{...}.

    Shared by the standalone add kernel and the MSM bucket loop. The statement
    order is chosen for liveness; see gen_g1_add.
    """
    L = []
    L += mont_mul("ZA", a + "Z", a + "Z", ind)
    L += mont_mul("ZB", b + "Z", b + "Z", ind)
    L += mont_mul("UA", a + "X", "ZB", ind)
    L += mont_mul("UB", b + "X", "ZA", ind)
    L += mont_mul("T1", a + "Y", b + "Z", ind)
    L += mont_mul("S1", "T1", "ZB", ind)
    L += mont_mul("T1", b + "Y", a + "Z", ind)
    L += mont_mul("S2", "T1", "ZA", ind)
    L += sub_mod("HH", "UB", "UA", ind)
    L += add_mod("T1", a + "Z", b + "Z", ind)
    L += mont_mul("T2", "T1", "T1", ind)
    L += sub_mod("T1", "T2", "ZA", ind)
    L += sub_mod("T2", "T1", "ZB", ind)
    L += mont_mul(dst + "Z", "T2", "HH", ind)
    L += dbl_mod("T1", "HH", ind)
    L += mont_mul("II", "T1", "T1", ind)
    L += mont_mul("JJ", "HH", "II", ind)
    L += sub_mod("T1", "S2", "S1", ind)
    L += dbl_mod("RR", "T1", ind)
    L += mont_mul("VV", "UA", "II", ind)
    L += mont_mul("T1", "RR", "RR", ind)
    L += sub_mod("T2", "T1", "JJ", ind)
    L += dbl_mod("T1", "VV", ind)
    L += sub_mod(dst + "X", "T2", "T1", ind)
    L += sub_mod("T1", "VV", dst + "X", ind)
    L += mont_mul("T2", "RR", "T1", ind)
    L += mont_mul("T1", "S1", "JJ", ind)
    L += dbl_mod("UA", "T1", ind)
    L += sub_mod(dst + "Y", "T2", "UA", ind)
    return L


def gen_msm_bucket():
    """Pippenger bucket accumulation: one thread per bucket.

    The host bins point indices by scalar window digit and hands over a CSR
    pair - `Off[b]..Off[b+1]` names the slice of `Idx` belonging to bucket `b`.
    Each thread walks its own slice, so the scatter that makes bucket
    accumulation awkward on a GPU (many scalars landing in one bucket) is
    resolved on the host by a counting sort, which is cheap integer work.

    The accumulator is SEEDED with the first point of the slice rather than
    with the identity, because `add-2007-bl` cannot represent the identity -
    it has no zero element. Empty buckets (`Off[b] == Off[b+1]`) therefore
    produce garbage here and the host treats them as the identity; it already
    knows which they are. The loop is pre-guarded (`setp.ge` + branch, not a
    do-while), so an empty slice runs zero times rather than once.

    `Idx` carries its OWN length, `NIdx`, separate from `NPts`. Those two are
    equal only for a single window; a whole MSM bins every point once per
    window, so `NIdx = NPts * num_windows` and the bucket array covers every
    window at once (`bucket = window * 2^c + digit`). Sharing one length
    parameter would have capped a launch at one window - 2^c threads, which is
    256 of them at c=8, i.e. a GPU running one warp per SM.

    The invariant is `k >= ks`, not `k >= 0`: `ks` is a runtime value, so
    `k >= 0` is genuinely unprovable and z3 says `sat` (a counterexample) when
    asked. See gotcha #11 - the checker was reporting that as "the solver
    could not be run", which is a different claim and sent the first
    investigation at the loop's SIZE rather than at its BOUNDS.
    """
    use_field(FQ)
    L = [HEADER.replace("%d", "N"), "",
         "// Pippenger bucket accumulation over BN254 G1, one thread per bucket.",
         "// Idx/Off are a CSR binning of point indices by scalar window digit.",
         "kernel bn254_msm_bucket(",
         "    PX: GlobalMemory<U32>,",
         "    PY: GlobalMemory<U32>,",
         "    PZ: GlobalMemory<U32>,",
         "    Idx: GlobalMemory<U32>,",
         "    Off: GlobalMemory<U32>,",
         "    RX: GlobalMemory<U32>,",
         "    RY: GlobalMemory<U32>,",
         "    RZ: GlobalMemory<U32>,",
         "    NB: I32,",
         "    NPts: I32,",
         "    NIdx: I32",
         ") {",
         "    // The block size is NOT hardcoded here. This kernel is",
         "    // latency-bound (NCU: 70% of stalls are `wait`, DRAM under 9%,",
         "    // SM under 27%), and at 136 registers a 256-thread CTA fits ONE",
         "    // block per SM -- 16.7% theoretical occupancy. Smaller blocks",
         "    // are the cheapest way to raise it, so the host chooses.",
         "    let b: I32 = block_idx_x() * block_dim_x() + thread_idx_x();",
         "    let nb1: I32 = NB + 1;",
         "    let s: U32 = block_ptr2d_load(Off, 0, b, nb1, 1, nb1);",
         "    let b1: I32 = b + 1;",
         "    let e: U32 = block_ptr2d_load(Off, 0, b1, nb1, 1, nb1);",
         "    let first: U32 = block_ptr2d_load(Idx, 0, s, NIdx, 1, NIdx);"]
    temps = ["ZA", "ZB", "UA", "UB", "S1", "S2", "HH", "II", "JJ", "RR", "VV", "T1", "T2"]
    check_names(temps)
    # accumulator
    L += load("aX", "PX", "first", "NPts")
    L += load("aY", "PY", "first", "NPts")
    L += load("aZ", "PZ", "first", "NPts")
    for v in temps + ["nX", "nY", "nZ", "bX", "bY", "bZ"]:
        L += [f"    let {v}{j}: U32 = 0;" for j in range(S)]
    L += scratch_decls()
    L += ["    let ks: U32 = s + 1;",
          "    @invariant(k >= ks)",
          "    for k in ks..e {",
          "        let pi: U32 = block_ptr2d_load(Idx, 0, k, NIdx, 1, NIdx);"]
    L += load("bX", "PX", "pi", "NPts", ind="        ")
    L += load("bY", "PY", "pi", "NPts", ind="        ")
    L += load("bZ", "PZ", "pi", "NPts", ind="        ")
    L += g1_add_body("n", "a", "b", ind="        ")
    for c in ["X", "Y", "Z"]:
        L += [f"        a{c}{j} = n{c}{j};" for j in range(S)]
    L += ["    }"]
    L += store("aX", "RX", "b", "NB")
    L += store("aY", "RY", "b", "NB")
    L += store("aZ", "RZ", "b", "NB")
    L += ["}", "", "fn main() {}", ""]
    use_field(FR)
    return "\n".join(L)


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name, text in [("tests/bn254_fr_mul_fast.ysu", gen_mul()),
                       ("tests/bn254_ntt_stage.ysu", gen_ntt()),
                       ("tests/bn254_ntt4_stage.ysu", gen_ntt4()),
                       ("tests/bn254_ntt4_fused.ysu", gen_ntt4_fused()),
                       ("tests/bn254_g1_add.ysu", gen_g1_add()),
                       ("tests/bn254_g1_dbl.ysu", gen_g1_dbl()),
                       ("tests/bn254_msm_bucket.ysu", gen_msm_bucket()),
                       ("tests/bn254_permute.ysu", gen_permute()),
                       ("tests/bn254_sub_vec.ysu", gen_sub_vec())]:
        path = os.path.join(here, name)
        with open(path, "w") as f:
            f.write(text)
        print(f"wrote {name}  ({len(text.splitlines())} lines)")
    for h in HIGH_DEPTHS:
        name = f"tests/bn254_ntt4_fused_high{h}.ysu"
        text = gen_ntt4_fused_high(h)
        with open(os.path.join(here, name), "w") as f:
            f.write(text)
        print(f"wrote {name}  ({len(text.splitlines())} lines)")
    print(f"p limbs  {[hex(x) for x in P32]}")
    print(f"n0'      {hex(NP)}")
