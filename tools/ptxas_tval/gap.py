"""What is ACTUALLY gating the tensor-core GEMMs -- the whole opcode gap, not the first refusal.

The doc records "the next feature in front of the GEMMs is `cp.async`, which is
where they now refuse".  That is a statement about which instruction the
executor reaches FIRST, and this file exists because that is not the same
question as which features would have to be modelled to validate one.  A
first-refusal reading has already been wrong once in this repository, in the
other direction: a crash in a census read as a missing feature.

TWO COLUMNS, and they answer different things.

  refuses-at   what the validator does today: execute until something is
               unmodelled, then stop.  Sound, and it sees exactly one opcode.

  full gap     a CENSUS: on an unmodelled opcode, record it and SKIP it.
               Skipping is unsound for validation -- the state afterwards is
               not the kernel's state -- so the gap set is a LOWER BOUND on
               the feature list and the non-opcode errors it drags behind it
               are contaminated by the skipping.  Reported separately and
               never mixed into the opcode census.

TWO WAYS THIS CENSUS UNDER-REPORTS, both measured rather than supposed.

  A PREDICATED instruction whose predicate name the executor does not
  recognise is attributed to the PREDICATE, not to the opcode behind it.  So
  `@%rt_p0 bra $L;` is counted as `@%rt_p0` and the `bra` is invisible.  That
  hides control flow in the six coprocessor kernels (`%rt_p0`/`%qp0`).

  A SETUP failure yields an EMPTY opcode gap, because no instruction ever
  executes.  That reads as "nothing unmodelled" and sorts to the top of a cost
  ranking.  Five kernels are in that state (`2 .shared arrays`): four
  `gemm_fp8_*` and `rmsnorm_residual_4096`.  `--rank` flags them; the
  per-kernel output shows it only in the `first=` column, easy to skim past.

  The two together are why `bra` reads as 37 kernels where a textual scan says
  48: 6 hidden by predication, 5 by setup failure.  Dropping predication alone
  from the textual scan gives 42, not 37 -- which is how the split was
  measured rather than assumed.  `loopgap.has_control_flow` scans the text.

`--rank` adds the aggregation the per-kernel output cannot give: cost per
kernel AND reach per opcode.  See `rank()`.
"""
import sys, re, glob, os
from z3 import *
import ptxexec, sassexec, smem, batch, params, mulmode

OPCODE_ERR = re.compile(r'UNMODELLED (?:PTX|SASS) OPCODE')

def fresh(ptxf):
    """The DRIVER's vocabulary, not a hand-written one.

    A census built on its own symbol set measures the census's symbol set.
    `batch.mk` is what `tval.run` uses, so a refusal here is a refusal there."""
    _, layout = params.parse(ptxf)
    return batch.mk(mulmode.MODES['wide'](), layout)

def ptx_insns(path):
    out, started = [], False
    for line in open(path):
        s = line.strip()
        if s.startswith('//') or not s: continue
        if s.startswith('.') or s.endswith('(') : continue
        if s == '{': started = True; continue
        if s == '}': break
        if not started or not s.endswith(';'): continue
        out.append(s[:-1].strip())
    return out

def sass_insns(path):
    out = []
    for line in open(path).read().splitlines():
        m = sassexec.INSN.match(line)
        if m: out.append((int(m.group(1),16), m.group(2).strip()))
    return out

def census(kernel):
    p, s = f'corpus/{kernel}.ptx', f'corpus/{kernel}.sass'
    r = {'kernel': kernel}
    for side in ('ptx','sass'):
        sym = fresh(p)
        first, opgap, other = None, [], []
        try:
            if side == 'ptx':
                sym.setdefault('smem_layout', {}).update(smem.layout(p))
                st = ptxexec.Ptx(sym); items = [(None,i) for i in ptx_insns(p)]
            else:
                text = open(s).read()
                st = sassexec.Sass(sym)
                st.labels = {m.group(1): int(m.group(2),16)
                             for m in re.finditer(r'\.L_(\w+):\s*\n\s*/\*([0-9a-f]+)\*/', text)}
                st.joins = []
                items = sass_insns(s)
        except Exception as e:
            r[side] = ('setup: '+str(e)[:60], [], []); continue
        for addr, ins in items:
            try:
                if side == 'ptx': st.step(ins)
                else: st.arrive(addr); st.step(ins, addr)
            except Exception as e:
                m = str(e)
                if first is None: first = m.split('\n')[0][:70]
                if OPCODE_ERR.search(m):
                    op = m.split("'")[1] if "'" in m else m[:30]
                    if op not in opgap: opgap.append(op)
                else:
                    k = m.split('\n')[0][:50]
                    if k not in other: other.append(k)
        r[side] = (first, opgap, other)
    return r

def rank(rows):
    """Two rankings of one census, because they answer different questions.

    COST  is what it takes to validate a given kernel.  It is the number the
          roadmap has always quoted, and on its own it picks the CHEAPEST
          kernel, which is not the same as a useful one.
    REACH is how many kernels an opcode blocks.  Nothing had computed it, and
          it re-orders the queue: `ptx_subword_ops` is the cheapest kernel in
          the corpus (2 PTX + 1 SASS opcodes, no contaminated errors) and every
          one of those three blocks exactly ONE kernel, so closing it buys
          1/66 and no leverage.

    A SETUP FAILURE IS FLAGGED, NEVER SHOWN AS A GAP OF ZERO.  When the census
    cannot build the initial state it executes no instruction, so it reports an
    empty opcode set -- which reads as "nothing unmodelled" and sorts to the
    top of a cost ranking.  Five corpus kernels are in exactly that state
    (`2 .shared arrays`), and an unflagged ranking calls them the cheapest work
    available."""
    import collections
    reach = collections.Counter()
    cost = []
    for k, r in rows:
        gp, gs = r['ptx'][1], r['sass'][1]
        setup = [side for side in ('ptx', 'sass') if str(r[side][0]).startswith('setup:')]
        for o in gp + gs:
            reach[o] += 1
        cost.append((len(gp) + len(gs), k, gp, gs, setup))
    cost.sort()
    print('=== COST: kernels by total opcode gap ===')
    for tot, k, gp, gs, setup in cost:
        tag = f'  [SETUP FAILURE on {"+".join(setup)} -- this gap is not a measurement]' if setup else ''
        print(f'{tot:3d}  {k}{tag}')
        if gp: print(f'       ptx : {", ".join(gp)}')
        if gs: print(f'       sass: {", ".join(gs)}')
    print('\n=== REACH: opcodes by number of kernels blocked ===')
    for o, n in reach.most_common():
        print(f'{n:3d}  {o}')
    return cost, reach


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--rank']
    want_rank = '--rank' in sys.argv[1:]
    ks = args or sorted(os.path.basename(x)[:-4] for x in glob.glob('corpus/*.ptx'))
    rows = []
    for k in ks:
        r = census(k)
        rows.append((k, r))
        if not want_rank:
            for side in ('ptx','sass'):
                first, opgap, other = r[side]
                print(f'{k:34s} {side:4s} first={first}')
                print(f'{"":34s}      opcode gap ({len(opgap)}): {", ".join(opgap) if opgap else "-"}')
                if other: print(f'{"":34s}      other ({len(other)}, contaminated): {"; ".join(other[:3])}')
    if want_rank:
        # FLOOR.  A census that examined nothing ranks nothing, perfectly.
        if not rows:
            print('FAIL: examined no kernels -- there is nothing to rank'); sys.exit(1)
        rank(rows)
