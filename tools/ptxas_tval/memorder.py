"""The GLOBAL memory model's preconditions -- checked, not assumed.

Both executors read every global load from ONE initial array, `mem`, and record
global stores as a trace that the validators pair BY ADDRESS, in any order.
That model is exact under conditions nobody had written down as a check:

  (1) NO GLOBAL LOAD FOLLOWS A GLOBAL STORE, on either side.  In the model a
      load reads the INITIAL memory whatever was stored before it; on the
      machine it reads the store whenever the two addresses meet.  So a
      translation that hoists a load above a store it could read back -- or
      sinks it below one -- builds the same terms on both sides and VALIDATES.
      `ptxexec` stated this as a fact about kernels ("a kernel never reads back
      what it wrote to global memory in the same launch").  It is a fact about
      the standing results, measured; as an ordering property it is false of
      kernels in the corpus, `y_cpu_matmul` among them.  (No count here: it came
      from a one-off scan, and a number in a comment is not a check.)

  (2) A STORE THE PAIRING REORDERS DOES NOT OVERLAP ONE IT CROSSES.  Stores are
      matched by address equality, so a translation that swaps two stores is
      accepted however they overlap -- and two stores through two pointers the
      host may alias, or through one pointer at +0 and +3, end with different
      bytes in memory when their order changes.

(1) is a REFUSAL: the model cannot represent a read-back, so a kernel that has
one is refused by name rather than validated against the wrong memory.  (2) is
an OBLIGATION, discharged per reordered pair: if both stores happen, their
4-byte extents are disjoint.  Where a validator's pairing is the identity it
adds nothing, so a result that does not reorder is unchanged -- which is every
standing result, measured.

THE ORDER IS RECORDED BY THE TRACE ITSELF, not by each executor arm.  A store
appended at an arm that forgot to log its position would make (1) vacuous for
exactly that opcode -- the one-site bug this repository keeps meeting -- so the
`loads`/`stores` lists are replaced by a list type that records on `append`
and refuses every other mutation.
"""
from z3 import BitVecVal, UGE, And, Implies, is_bv, is_bool

# Every global store the executors record is one 32-bit word (a v4 store is
# four of them, `STG.E.64` two).  The overlap obligation is stated for that
# width, so a store of any other width is refused rather than reasoned about.
STORE_WIDTH_BYTES = 4


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


def require_no_read_back(side, regions):
    """Precondition (1) for one side.  `regions` in execution order; a loop body
    is passed TWICE so that a store in one iteration followed by a load in the
    next is seen."""
    seq = order_of(regions)
    first = seq.find('S')
    if first >= 0 and 'L' in seq[first:]:
        n = seq[first:].count('L')
        raise Refusal(
            f'{side}: {n} global load(s) follow a global store; the memory model reads '
            f'every global load from the INITIAL array, so a load that could read back an '
            f'earlier store is a program it cannot represent  (refusing, not guessing)')


def reorder_obligations(stores, perm):
    """Precondition (2).  `stores` are one side's (addr, value, guard) in its own
    program order and `perm[i]` is where store i landed on the other side.

    Returns ((i, j), claim) for every pair the permutation puts in the opposite
    order.  The claim is: if both stores happen, their extents are disjoint --
    `a_j - a_i` lies in [4, 2^64 - 4] modulo 2^64, written as two unsigned
    comparisons so that wrap-around is part of it rather than an exception."""
    for a, v, g in stores:
        if not (is_bv(a) and a.size() == 64 and is_bv(v) and v.size() == 8 * STORE_WIDTH_BYTES
                and is_bool(g)):
            raise Refusal(f'memorder: a store is not a 64-bit address, a '
                          f'{8 * STORE_WIDTH_BYTES}-bit word and a Boolean guard; the overlap '
                          f'obligation is stated for that shape only  (refusing, not guessing)')
    if sorted(perm) != list(range(len(stores))):
        raise Refusal(f'memorder: the store pairing {perm} is not a permutation  '
                      f'(refusing, not guessing)')
    obs = []
    w = BitVecVal(STORE_WIDTH_BYTES, 64)
    for i in range(len(stores)):
        for j in range(i + 1, len(stores)):
            if perm[i] > perm[j]:
                ai, _vi, gi = stores[i]
                aj, _vj, gj = stores[j]
                d = aj - ai
                obs.append(((i, j), Implies(And(gi, gj), And(UGE(d, w), UGE(-d, w)))))
    return obs


def _self_check():
    """Pin the recorder AT IMPORT.  Every standing result has an identity
    pairing and no read-back, so no behavioural gate can see these helpers
    broken on the corpus it runs on; the fixtures in `mem/` can, and this makes
    a broken helper fail loudly in every tool that imports it as well."""
    class _S: pass
    s = _S(); install(s)
    s.loads.append(0); s.stores.append(0); s.loads.append(0)
    if s.mem_order != ['L', 'S', 'L']:
        raise Exception(f'memorder: the trace recorded {s.mem_order}, not [L, S, L]')
    try:
        require_no_read_back('X', [s])
        raise Exception('memorder: a load after a store was not refused')
    except Refusal:
        pass
    t = _S(); install(t)
    t.loads.append(0); t.loads.append(0); t.stores.append(0); t.stores.append(0)
    require_no_read_back('X', [t])          # loads first: must NOT refuse
    try:
        require_no_read_back('X', [t, t])   # a body run twice: S then L across it
        raise Exception('memorder: a store followed by a load in the next iteration '
                        'was not refused')
    except Refusal:
        pass
    try:
        t.stores.extend([0])
        raise Exception('memorder: extend bypassed the order record')
    except Refusal:
        pass
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


_self_check()
