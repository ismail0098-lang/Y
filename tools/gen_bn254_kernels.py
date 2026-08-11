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

P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
P32 = [(P >> (32 * i)) & 0xFFFFFFFF for i in range(8)]
NP = (-pow(P, -1, 1 << 32)) % (1 << 32)   # -p^-1 mod 2^32
assert (P * NP + 1) % (1 << 32) == 0
S = 8   # limbs


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


def load(var, buf, row, stride, maxr, ind="    "):
    return [f"{ind}let {var}{j}: U32 = block_ptr2d_load({buf}, {row}, {j}, "
            f"{stride}, {maxr}, {stride});" for j in range(S)]


def store(var, buf, row, stride, maxr, ind="    "):
    return [f"{ind}block_ptr2d_store({buf}, {row}, {j}, {stride}, {maxr}, "
            f"{stride}, {var}{j});" for j in range(S)]


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
    L += load("a", "A", "tid", 8, "N")
    L += load("b", "B", "tid", 8, "N")
    L += [f"    let r{j}: U32 = 0;" for j in range(S)]
    L += scratch_decls()
    L += mont_mul("r", "a", "b")
    L += store("r", "Out", "tid", 8, "N")
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
    L += load("u", "X", "i0", 8, "NElem")
    L += load("q", "X", "i1", 8, "NElem")
    L += load("w", "Tw", "pos", 8, "Half")
    L += [f"    let v{j}: U32 = 0;" for j in range(S)]
    L += [f"    let s{j}: U32 = 0;" for j in range(S)]
    L += [f"    let e{j}: U32 = 0;" for j in range(S)]
    L += scratch_decls()
    L += ["    // v = x[i1] * w  (Montgomery: w carries the R factor)"]
    L += mont_mul("v", "q", "w")
    L += ["    // s = u + v,  e = u - v"]
    L += add_mod("s", "u", "v")
    L += sub_mod("e", "u", "v")
    L += store("s", "X", "i0", 8, "NElem")
    L += store("e", "X", "i1", 8, "NElem")
    L += ["}", "", "fn main() {}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name, text in [("tests/bn254_fr_mul_fast.ysu", gen_mul()),
                       ("tests/bn254_ntt_stage.ysu", gen_ntt())]:
        path = os.path.join(here, name)
        with open(path, "w") as f:
            f.write(text)
        print(f"wrote {name}  ({len(text.splitlines())} lines)")
    print(f"p limbs  {[hex(x) for x in P32]}")
    print(f"n0'      {hex(NP)}")
