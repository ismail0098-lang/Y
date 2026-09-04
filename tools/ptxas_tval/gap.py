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

if __name__ == '__main__':
    ks = sys.argv[1:] or sorted(os.path.basename(x)[:-4] for x in glob.glob('corpus/*.ptx'))
    for k in ks:
        r = census(k)
        for side in ('ptx','sass'):
            first, opgap, other = r[side]
            print(f'{k:34s} {side:4s} first={first}')
            print(f'{"":34s}      opcode gap ({len(opgap)}): {", ".join(opgap) if opgap else "-"}')
            if other: print(f'{"":34s}      other ({len(other)}, contaminated): {"; ".join(other[:3])}')
