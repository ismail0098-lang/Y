"""Shared memory, and what a barrier means to a PER-THREAD equivalence check.

## The model

Shared memory is a z3 Array from 32-bit WORD index to 32-bit value, one array
shared by both executors, threaded through the state.  Word-indexed because
every shared access this validator can reach is 4-byte granular (`ld.shared.v4`
/ `LDS.128` and their scalar forms); the sub-word shared traffic in the fp8
GEMMs (`ld.shared.u8`, `STS.U16`) is REFUSED BY NAME rather than approximated,
because a byte lane packed into a word model is the `ld.global.f32`-on-a-u32-
buffer bug wearing a different address space.

## The barrier, which is the whole design

The tempting model is that `bar.sync` is a no-op: one thread's shared memory is
just its own array, and a barrier does not change it.  That is WRONG in the way
this project keeps finding -- it is not conservative, it VALIDATES.  With a
no-op barrier every shared-memory kernel passes trivially, including one where
ptxas has hoisted a store from before the barrier to after it, which is the
single most valuable thing a barrier check could catch.  Same shape as modelling
a PTX macro-op and the MUFU that seeds it as one uninterpreted function.

The sound model is that after a barrier the contents of shared memory are an
unknown function of the contents before it -- unknown because they now depend on
every other thread, a function because those other threads are running the same
program:

    smem_after = H_k(smem_before)

with `H_k` uninterpreted, the SAME `H_k` on both sides at the same barrier
index.  That gives exactly what is wanted, by congruence alone:

  - equal writes before the barrier  =>  equal arrays  =>  equal reads after it.
    So a faithful translation validates, with no axiom about other threads.
  - a store MOVED ACROSS the barrier changes `H_k`'s ARGUMENT, so nothing forces
    the results equal and the validator reports it.
  - a store DROPPED or INVENTED before the barrier, likewise.
  - a barrier dropped or invented changes the number of applications, so the
    indices no longer pair; refused before any query is posed.

It is sound for per-thread equivalence in the ordinary sense: every real
execution of the block induces SOME function from pre-barrier to post-barrier
contents, so any concrete behaviour is an instance of `H_k`.  It is
deliberately not a proof about the block's cross-thread behaviour, and this
validator does not claim one -- what it claims is that ptxas did not change the
program, which is the whole thesis.

## Addressing

PTX names shared memory by SYMBOL plus offset (`mov.u64 %rd1, __y_smem_0`);
SASS uses a raw byte offset into the window, so the symbol's base is the offset
ptxas assigned it.  With one `.shared` array that is 0.  With more than one it
is a layout decision made inside ptxas that nothing in the artifact states, so
MORE THAN ONE SHARED SYMBOL IS REFUSED: guessing the second base would alias two
arrays onto each other and validate a kernel that swaps them.
"""
from z3 import *

W = 32
SMEM_SORT = ArraySort(BitVecSort(W), BitVecSort(W))


def fresh():
    """The initial contents: unconstrained, and the SAME term on both sides.

    Shared memory is not zeroed at block start, so reading before writing is
    undefined; an unconstrained array is the faithful model of that and makes
    such a read equal on both sides only if both read the same address.
    """
    return Array('smem0', BitVecSort(W), BitVecSort(W))


class Barriers:
    """The `H_k` family, memoised so both executors get the same function.

    Lives in `sym` rather than in either executor, for the reason the multiply
    primitive does: two sides that build their own would share no structure and
    congruence could not relate them.
    """

    def __init__(self):
        self.fns = {}
        self.ptx_count = 0
        self.sass_count = 0

    def h(self, k):
        if k not in self.fns:
            self.fns[k] = Function(f'bar{k}', SMEM_SORT, SMEM_SORT)
        return self.fns[k]

    def apply(self, arr, side):
        """Cross barrier number `k` on `side`, returning the new contents."""
        k = self.ptx_count if side == 'ptx' else self.sass_count
        if side == 'ptx':
            self.ptx_count += 1
        else:
            self.sass_count += 1
        return self.h(k)(arr)

    def agree(self):
        return self.ptx_count == self.sass_count


def _win(byte_addr):
    """Truncate a byte address to the 32-bit shared window.

    THE SHARED WINDOW IS 32-BIT ADDRESSED AND PTX ADDRESSES IT WITH 64-BIT
    REGISTERS, so the truncation has to happen BEFORE the word shift, not after.
    Getting that order wrong is invisible on every in-range access and wrong on
    exactly the wrapping ones: `smem_roundtrip` computes `(255 - tid) << 4` via
    `cvt.u64.u32` + `shl.b64`, which for tid > 255 zero-extends a near-2^32
    value and shifts it in SIXTY-FOUR bits with no wrap, while SASS's
    `[R0.X16]` scales in 32 and wraps.  Shift-then-truncate makes those two
    genuinely different numbers (17179868140 against 1073740780) and reports a
    correct kernel as a mismatch; truncate-then-shift makes them agree, which is
    what the hardware does.

    Found by a counterexample at tid_x = 516 rather than by reading the ISA
    manual, and it is the ordinary shape of a modelling bug here: both readings
    are plausible, they agree everywhere a test would look, and only the
    out-of-block path separates them.
    """
    return byte_addr if byte_addr.size() == W else Extract(W - 1, 0, byte_addr)


def word(byte_addr):
    """Byte address -> word index within the 32-bit shared window.

    The low two bits must be zero.  NOT masked away: an unaligned shared access
    is a program this validator does not model, and silently rounding it down
    would answer a different question.  The check is a proof obligation rather
    than an assert because the address is symbolic; callers pass it to
    `require_aligned`.
    """
    return LShR(_win(byte_addr), 2)


def require_aligned(byte_addr):
    """The obligation that a shared access is 4-byte aligned."""
    return Extract(1, 0, _win(byte_addr)) == BitVecVal(0, 2)


def refuse_subword(op):
    raise Exception(
        f'UNMODELLED SHARED ACCESS {op!r}: shared memory is modelled as 32-bit '
        f'words, and packing a byte lane into that model is the same bug as '
        f'loading a u32 buffer with ld.global.f32  (refusing, not guessing)')


def layout(ptx_path):
    """Base byte offset of each `.shared` symbol, from the PTX declarations.

    Returns {name: base}.  Refuses more than one -- see the module docstring.
    """
    import re
    decls = []
    for ln in open(ptx_path):
        # FOUR FORMS OCCUR IN THIS CORPUS AND THE UNSIZED ONE IS THE COMMONEST.
        #   .shared .align 16 .b32 __y_smem_0[2048]       static, word-typed
        #   .shared .align 4  .b8  smem_fp8_A[8192]       static, byte-typed
        #   .extern .shared .align 16 .b8 smem_pipe[]     DYNAMIC, no size
        # A regex requiring `[<digits>]` silently misses the third, which is 23
        # of the corpus's 33 shared-memory kernels -- the census reported them as
        # an unmodelled OPERAND, i.e. as a missing opcode rather than a missing
        # declaration form, which is why it read as a bigger gap than it was.
        m = re.match(r'\s*(\.extern\s+)?\.shared\s+\.align\s+(\d+)\s+\.b(\d+)\s+(\w+)\[(\d*)\]', ln)
        if m:
            ext, align, bits, name, n = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
            if ext and n:
                raise Exception(f'UNMODELLED: .extern .shared {name} declares a SIZE; '
                                f'dynamic shared memory is sized at launch  (refusing, not guessing)')
            decls.append((name, align, (bits * int(n) // 8) if n else None))
    if len(decls) > 1:
        raise Exception(
            f'UNMODELLED: {len(decls)} .shared arrays ({", ".join(d[0] for d in decls)}). '
            f'Their bases are a ptxas layout decision the artifact does not state, and '
            f'guessing the second one aliases two arrays  (refusing, not guessing)')
    return {d[0]: 0 for d in decls}
