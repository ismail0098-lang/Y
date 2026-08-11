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


def mont_mul(dst, a, b, ind="    "):
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


def add_mod(dst, a, b, ind="    "):
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


def sub_mod(dst, a, b, ind="    "):
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


def scratch_decls(ind="    "):
    L = [f"{ind}// Scratch, declared once so unrolling does not allocate a",
         f"{ind}// register per step.  `pr` and `sw` are the 64-bit lanes that",
         f"{ind}// make carries visible: no operator exposes them.",
         f"{ind}let pr: U64 = 0;", f"{ind}let sw: U64 = 0;",
         f"{ind}let c: U32 = 0;", f"{ind}let nc: U32 = 0;",
         f"{ind}let m: U32 = 0;",
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
    knows which they are.

    STATUS: this generates and parses, and it does NOT currently compile.
    `@safe`'s invariant checker refuses it:

        [Strict Safety] Could not verify invariant `(k >= 0)` (initiation
        check) because the SMT solver could not be run.

    z3 is installed and present (`venv/bin/z3`), and the same invariant on the
    same shape of loop verifies fine in `tests/bn254_fr_mul.ysu`. What is
    different here is SIZE: the loop body is a full 30-multiply point
    addition, ~9,000 lines of Y, and `trace_body_statements` encodes the whole
    body into one SMT query per obligation. The solver does not come back.

    This is a real limitation of the verification layer meeting generated
    code, not a property of the kernel, and the fix belongs in
    `type_checker.rs` - the body of a loop whose invariant mentions only the
    loop variable does not need encoding at all, and more generally the
    encoder should slice the body to statements that can affect the invariant.

    It is NOT fixed by `Y_ALLOW_UNVERIFIED_INVARIANTS=1`. CLAUDE.md says so
    directly and it is right: that variable does not make the check pass, it
    makes the check not happen.
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
         "    NPts: I32",
         ") {",
         "    let b: I32 = block_idx_x() * 256 + thread_idx_x();",
         "    let nb1: I32 = NB + 1;",
         "    let s: U32 = block_ptr2d_load(Off, 0, b, nb1, 1, nb1);",
         "    let b1: I32 = b + 1;",
         "    let e: U32 = block_ptr2d_load(Off, 0, b1, nb1, 1, nb1);",
         "    let first: U32 = block_ptr2d_load(Idx, 0, s, NPts, 1, NPts);"]
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
          "    @invariant(k >= 0)",
          "    for k in ks..e {",
          "        let pi: U32 = block_ptr2d_load(Idx, 0, k, NPts, 1, NPts);"]
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
                       ("tests/bn254_g1_add.ysu", gen_g1_add()),
                       ("tests/bn254_g1_dbl.ysu", gen_g1_dbl()),
                       ("tests/bn254_msm_bucket.ysu", gen_msm_bucket())]:
        path = os.path.join(here, name)
        with open(path, "w") as f:
            f.write(text)
        print(f"wrote {name}  ({len(text.splitlines())} lines)")
    print(f"p limbs  {[hex(x) for x in P32]}")
    print(f"n0'      {hex(NP)}")
