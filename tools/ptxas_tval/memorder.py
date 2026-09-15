"""The GLOBAL memory model: store-ordered and byte-faithful.

WHAT IT WAS.  Both executors read every global load from ONE initial array,
`mem`, and recorded global stores as a trace the validators pair BY ADDRESS, in
any order.  That is exact under two conditions nobody had written down, and the
previous version of this file checked them:

  (1) NO GLOBAL LOAD FOLLOWS A GLOBAL STORE, on either side.  In that model a
      load read the INITIAL memory whatever was stored before it; on the machine
      it reads the store whenever the two extents meet.  So a translation that
      hoists a load above a store it could read back -- or sinks it below one --
      built the same terms on both sides and VALIDATED.  It was a REFUSAL.

  (2) A STORE THE PAIRING REORDERS DOES NOT OVERLAP ONE IT CROSSES.  Stores are
      matched by address equality, so a translation that swaps two stores was
      accepted however they overlap.  It is an OBLIGATION, discharged per
      reordered pair, and it still is.

The refusal had a price stated as a standing row: `mem/lsls` is ptxas's CORRECT
output -- it kept a store above a load it could not prove unaliased -- and it
was refused, because a model that reads every load from the initial array cannot
tell it from the wrong `mem/las_ptx`.

WHAT IT IS.  (1) is MODELLED.  A global load reads memory as updated by every
global store that PRECEDES it on its own side, byte by byte (`read_through`), so
a load hoisted above a store it could read back builds a DIFFERENT term from the
one below it, and the store value that load feeds is refuted rather than refused.
The straight-line validators no longer refuse a read-back.  `loopval` still
refuses one that crosses a REGION boundary (`require_no_read_back_across`),
because it executes regions separately and a region's store trace starts empty.

BYTE-FAITHFUL, AND THE WIDTH IS THE STORE'S OWN.  A store is (64-bit address,
value, guard) and its width in bytes is the VALUE's width -- 32 for a word, 8 or
16 for a sub-word store.  The executors' base convention is unchanged: `base`, a
32-bit word, stands for the initial bytes at addr..addr+3, little-endian, which
is what "the word at a byte address" means.  A partial overlap therefore takes
the bytes it does not cover from `base`, and nothing has to be invented.  The
earlier model had no width, which is why a sub-word store was correctly refused
until now: comparing it by address alone could not tell an 8-bit store from a
32-bit one at the same address.

NO STORE, NO NEW TERM.  With nothing stored yet `read_through` returns `base`
itself and builds no z3 node.  Every load in every standing result precedes
every store -- measured, both sides, every row -- so those results build exactly
the terms they did before, which is checked by fingerprinting the terms and not
argued.

THE ORDER IS RECORDED BY THE TRACE ITSELF, not by each executor arm.  A store
appended at an arm that forgot to log its position would make the order vacuous
for exactly that opcode -- the one-site bug this repository keeps meeting -- so
the `loads`/`stores` lists are a list type that records on `append` and refuses
every other mutation.  And a load cannot bypass `read_through`: each executor
has ONE method that touches `sym['mem']`, pinned at import (`pin_one_memory_path`).
"""
import inspect
from z3 import (BitVecVal, UGE, ULT, And, Implies, If, Extract, Concat,
                is_bv, is_bool)

# A global store is 1, 2 or 4 bytes wide, by its value's width.  Anything else
# is refused rather than reasoned about.
STORE_WIDTHS_BYTES = (1, 2, 4)


class Refusal(Exception):
    """A program the memory model cannot represent.  Reported as REFUSED."""


class _Trace(list):
    """A load or store list that records program order into a shared sequence."""

    def __init__(self, tag, order):
        super().__init__()
        self._tag, self._order = tag, order

    def append(self, item):
        self._order.append(self._tag)
        super().append(item)

    def _grow_only(self, *a, **k):
        raise Refusal('memorder: a load/store trace may only grow by append; any other '
                      'mutation would bypass the program-order record  (refusing, not guessing)')

    extend = insert = pop = remove = clear = sort = reverse = _grow_only
    __iadd__ = __setitem__ = __delitem__ = _grow_only


def install(st):
    """Give an executor state order-recording `loads`/`stores`."""
    st.mem_order = []
    st.loads = _Trace('L', st.mem_order)
    st.stores = _Trace('S', st.mem_order)


def order_of(regions):
    """The concatenated L/S sequence of regions given in EXECUTION order."""
    out = []
    for r in regions:
        if not hasattr(r, 'mem_order'):
            raise Refusal('memorder: a region was executed without an order record; '
                          'its loads and stores cannot be placed  (refusing, not guessing)')
        out += r.mem_order
    return ''.join(out)


def store_width(store):
    """A store's width in bytes, refusing a shape the model does not state."""
    a, v, g = store
    if not (is_bv(a) and a.size() == 64 and is_bv(v) and is_bool(g)
            and v.size() % 8 == 0 and v.size() // 8 in STORE_WIDTHS_BYTES):
        raise Refusal(f'memorder: a global store is not a 64-bit address, a value of '
                      f'{"/".join(str(8 * w) for w in STORE_WIDTHS_BYTES)} bits and a '
                      f'Boolean guard; the memory model is stated for that shape only  '
                      f'(refusing, not guessing)')
    return v.size() // 8


def _byte_of(v, d):
    """Byte `d` of store value `v`, little-endian; `d` is a 64-bit term the
    caller has already constrained below v's width."""
    r = Extract(7, 0, v)
    for j in range(1, v.size() // 8):
        r = If(d == BitVecVal(j, 64, d.ctx), Extract(8 * j + 7, 8 * j, v), r)
    return r


def read_through(stores, addr, base, nbytes=4):
    """What a load of `nbytes` bytes at `addr` reads, given the global stores that
    PRECEDE it in program order and `base`, a 32-bit word standing for the memory
    at addr..addr+3 before any store ran.

    Returns a 32-bit word.  Its low `nbytes` bytes are read through the stores,
    a later store wrapping an earlier one so the later one wins; its high bytes
    are `base`'s, and a sub-word load discards them.

    See the module docstring for why an empty `stores` must return `base` itself.
    """
    if not stores:
        return base
    if not (is_bv(addr) and addr.size() == 64 and is_bv(base) and base.size() == 32
            and nbytes in STORE_WIDTHS_BYTES):
        raise Refusal('memorder: a global load after a store is not a 64-bit address '
                      'reading 1, 2 or 4 bytes of a 32-bit word  (refusing, not guessing)')
    widths = [store_width(s) for s in stores]
    c = addr.ctx
    out = []
    for k in range(4):
        b = Extract(8 * k + 7, 8 * k, base)
        if k < nbytes:
            y = addr if k == 0 else addr + BitVecVal(k, 64, c)
            for (s, v, g), w in zip(stores, widths):
                d = y - s
                b = If(And(g, ULT(d, BitVecVal(w, 64, c))), _byte_of(v, d), b)
        out.append(b)
    return Concat(out[3], out[2], out[1], out[0])


def require_no_read_back_across(side, regions):
    """A load that could read back a store made in an EARLIER REGION.

    For `loopval`, which executes each region in a fresh state: a region's store
    trace starts empty, so its loads cannot see a store another region made, and
    `read_through` is exact WITHIN a region only.  `regions` in execution order;
    a loop body is passed TWICE so that a store in one iteration followed by a
    load in the next is seen.  A read-back inside one region is not refused --
    the executor models it."""
    stored_before = False
    for r in regions:
        seq = order_of([r])
        if stored_before and 'L' in seq:
            n = seq.count('L')
            raise Refusal(
                f'{side}: {n} global load(s) in a region that runs after a region '
                f'containing a global store; this validator executes regions '
                f'separately, so a load that could read back a store made in an '
                f'earlier region is a program it cannot represent  (refusing, not guessing)')
        stored_before = stored_before or 'S' in seq


def reorder_obligations(stores, perm):
    """Precondition (2).  `stores` are one side's (addr, value, guard) in its own
    program order and `perm[i]` is where store i landed on the other side.

    Returns ((i, j), claim) for every pair the permutation puts in the opposite
    order.  The claim is: if both stores happen, their extents are disjoint --
    with d = a_j - a_i modulo 2^64, j does not start inside i (d >= w_i) and i
    does not start inside j (-d >= w_j), written as two unsigned comparisons so
    that wrap-around is part of it rather than an exception.  With both widths 4
    this is exactly the claim the word-only version built."""
    widths = [store_width(s) for s in stores]
    if sorted(perm) != list(range(len(stores))):
        raise Refusal(f'memorder: the store pairing {perm} is not a permutation  '
                      f'(refusing, not guessing)')
    obs = []
    # The word width is built ONCE, up front, whether or not a pair is reordered:
    # that is the node the word-only version built, in that order, and the
    # multiply primitive orders operands by z3 node id -- so a different
    # allocation here renumbers every term a validator builds afterwards.
    c = stores[0][0].ctx if stores else None
    w4 = BitVecVal(4, 64, c)
    for i in range(len(stores)):
        for j in range(i + 1, len(stores)):
            if perm[i] > perm[j]:
                ai, _vi, gi = stores[i]
                aj, _vj, gj = stores[j]
                d = aj - ai
                wi = w4 if widths[i] == 4 else BitVecVal(widths[i], 64, c)
                wj = w4 if widths[j] == 4 else BitVecVal(widths[j], 64, c)
                obs.append(((i, j), Implies(And(gi, gj), And(UGE(d, wi), UGE(-d, wj)))))
    return obs


def pair_by_address(hits):
    """A pairing from address matches, allowing EQUAL-ADDRESS GROUPS.

    `hits[i]` is the list of the other side's indices whose address was PROVED
    equal to access i's.  A unique match pairs as it always did.  Several
    accesses at one address used to be refused (`matched 2`), and under the old
    model that was right: a pairing chose which initial-memory symbol a load
    got, and a wrong choice was a wrong proof.  It is not a choice any more.  A
    load's base symbol stands for the initial memory AT AN ADDRESS -- one value,
    however many loads read it -- and `read_through` adds whatever was stored in
    between.  So the accesses sharing a hit set form a group, the group shares
    ONE base (`rep`), and its members pair in occurrence order, which for stores
    also keeps their relative order, so the reorder obligation sees any swap.

    Returns (perm, rep) or (None, reason).  A hit set that is not exactly the
    group's size, or that two groups share, is refused by reason as before."""
    groups = {}
    for i, h in enumerate(hits):
        groups.setdefault(frozenset(h), []).append(i)
    perm, rep, claimed = [None] * len(hits), [None] * len(hits), set()
    for hs, members in groups.items():
        if not hs or len(hs) != len(members) or hs & claimed:
            i = members[0]
            return None, f'{i} matched {len(hits[i])}'
        claimed |= hs
        for m, j in zip(members, sorted(hs)):
            perm[m], rep[m] = j, members[0]
    return perm, rep


def pin_one_memory_path(cls, method='gload'):
    """An executor class reads global memory in exactly one method.

    `read_through` is only faithful if EVERY global load goes through it.  An arm
    that wrote `Select(self.sym['mem'], addr)` itself would read the initial array
    for that one opcode -- the one-site bug -- and no fixture exercising a
    different opcode could see it.  So the class source is read AT IMPORT and the
    memory symbol may appear in `method` and nowhere else."""
    src = inspect.getsource(cls)
    body = inspect.getsource(getattr(cls, method))
    hits, hits_in = src.count("['mem']"), body.count("['mem']")
    if hits_in != 1 or hits != 1:
        raise Exception(f'memorder: {cls.__name__} touches the global memory symbol in '
                        f'{hits} place(s), {hits_in} of them in {method}; every global '
                        f'load must go through {method} and read_through  '
                        f'(a load that bypasses it reads the INITIAL memory)')


def _self_check():
    """Pin the model AT IMPORT.  Every standing result has an identity pairing
    and no read-back, so no behavioural gate over them can see these helpers
    broken; the fixtures in `mem/` can, and this makes a broken helper fail
    loudly in every tool that imports it as well.

    TWO CONTEXTS, AND WHY.  z3 numbers AST nodes as they are made and REUSES the
    number of a freed one, and the multiply primitive (and the float UFs) order
    their operands by that number.  So the nodes a self-check builds at import
    decide the operand order of terms a validator builds later in the same
    process.  Measured, not supposed: extending this check in the main context
    changed the term fingerprints of NINE of the sixteen standing rows without
    one executor line changing (with the check removed on both sides, 31 of 34
    rows were byte-identical and the other 3 were the read-back fixtures).  So
    the main-context section below is the one this file has always run, kept as
    it was, and every check added since runs in a PRIVATE `Context`, whose
    numbering nothing else shares."""
    class _S: pass
    s = _S(); install(s)
    s.loads.append(0); s.stores.append(0); s.loads.append(0)
    if s.mem_order != ['L', 'S', 'L']:
        raise Exception(f'memorder: the trace recorded {s.mem_order}, not [L, S, L]')
    require_no_read_back_across('X', [s])      # within one region: must NOT refuse
    t = _S(); install(t); t.loads.append(0); t.stores.append(0)
    try:
        require_no_read_back_across('X', [t, t])   # a body run twice
        raise Exception('memorder: a store followed by a load in the next iteration '
                        'was not refused')
    except Refusal:
        pass
    u = _S(); install(u); u.stores.append(0)
    v_ = _S(); install(v_); v_.loads.append(0)
    require_no_read_back_across('X', [v_, u])      # load region BEFORE store region: fine
    try:
        require_no_read_back_across('X', [u, v_])
        raise Exception('memorder: a load in a region after a storing region was not refused')
    except Refusal:
        pass
    try:
        t.stores.extend([0])
        raise Exception('memorder: extend bypassed the order record')
    except Refusal:
        pass
    # ---- the main-context section, unchanged in what it allocates ----------
    from z3 import BitVec, BoolVal, prove as _p, Solver, Not, unsat
    a = BitVec('memorder_a', 64); v = BitVec('memorder_v', 32)
    two = [(a, v, BoolVal(True)), (a + 4, v, BoolVal(True))]
    if reorder_obligations(two, [0, 1]):
        raise Exception('memorder: an identity pairing produced an obligation')
    ob = reorder_obligations(two, [1, 0])
    if len(ob) != 1:
        raise Exception('memorder: a swapped pair did not produce exactly one obligation')
    so = Solver(); so.add(Not(ob[0][1]))
    if so.check() != unsat:
        raise Exception('memorder: stores at +0 and +4 were not shown disjoint')
    three = [(a, v, BoolVal(True)), (a + 3, v, BoolVal(True))]
    so = Solver(); so.add(Not(reorder_obligations(three, [1, 0])[0][1]))
    if so.check() == unsat:
        raise Exception('memorder: stores at +0 and +3 were shown disjoint; the width is wrong')
    _private_self_check()


def _private_self_check():
    """Everything added with the byte-faithful model, in a private context."""
    from z3 import Context, BitVec, BoolVal, Solver, Not, unsat, simplify
    C = Context()
    a = BitVec('memorder_a', 64, C); v = BitVec('memorder_v', 32, C)
    b8 = BitVec('memorder_b', 8, C)
    T, F = BoolVal(True, C), BoolVal(False, C)
    lit = lambda n, w: BitVecVal(n, w, C)

    def disjoint(stores):
        so = Solver(ctx=C); so.add(Not(reorder_obligations(stores, [1, 0])[0][1]))
        return so.check() == unsat
    # mixed widths: a byte at +3 overlaps a word at +0; a byte at +4 does not; and
    # the SECOND width is the second store's, not a copy of the first
    if disjoint([(a, v, T), (a + lit(3, 64), b8, T)]):
        raise Exception('memorder: a byte at +3 was shown disjoint from a word at +0')
    if not disjoint([(a, v, T), (a + lit(4, 64), b8, T)]):
        raise Exception('memorder: a byte at +4 was not shown disjoint from a word at +0')
    if disjoint([(a + lit(3, 64), b8, T), (a, v, T)]):
        raise Exception('memorder: a word at +0 was shown disjoint from a byte at +3 '
                        '(the second width is not the second store\'s)')

    # read_through, both halves, on concrete addresses so the answer is a number:
    # an empty prefix must hand back the SAME node; a store must be seen; a later
    # store must win; a byte store must change one byte and only that byte.
    base = lit(0x44332211, 32); A = lit(0x1000, 64)
    if read_through([], A, base) is not base:
        raise Exception('memorder: read_through built a term with nothing stored '
                        '(standing results would no longer build the same terms)')
    cases = [
        ([(A, lit(0xDDCCBBAA, 32), T)], 4, 0xDDCCBBAA),
        ([(A, lit(0xDDCCBBAA, 32), F)], 4, 0x44332211),                    # guard false
        ([(A, lit(0xDDCCBBAA, 32), T), (A, lit(0x0000EEFF, 32), T)], 4, 0x0000EEFF),  # later wins
        ([(A + lit(1, 64), lit(0x99, 8), T)], 4, 0x44339911),              # one byte only
        ([(A - lit(2, 64), lit(0x88776655, 32), T)], 4, 0x44338877),       # partial, low
        ([(A + lit(2, 64), lit(0x7766, 16), T)], 1, 0x44332211),           # misses byte 0
        ([(A + lit(5, 64), lit(0x11223344, 32), T)], 4, 0x44332211),       # disjoint
    ]
    for stores, nb, want in cases:
        got = simplify(Extract(8 * nb - 1, 0, read_through(stores, A, base, nb))).as_long()
        want &= (1 << (8 * nb)) - 1
        if got != want:
            raise Exception(f'memorder: read_through read 0x{got:x} where the byte '
                            f'memory holds 0x{want:x}')


_self_check()
