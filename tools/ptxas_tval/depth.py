"""How DEEP is each blocker?  A first-refusal census names ONE opcode per kernel;
it says nothing about whether fixing that opcode advances the kernel or merely
moves the refusal one instruction along.  This lists the FULL opcode alphabet of
each kernel and marks every op that no kernel-passing-both-executors uses, so a
proposed unlock can be priced before it is built.

Static and approximate BY CONSTRUCTION: an opcode modelled but never exercised
by a passing kernel is reported as unknown.  It over-states the work, never
under-states it -- which is the safe direction for a go/no-go.
"""
import glob, os, re, sys, collections

PASSING = {'bn254_permute','bn254_sub_vec','ptx_carry_chain','bn254_fr_mul_fast',
           'bn254_g1_add','bn254_g1_dbl'}

def ptx_ops(f):
    ops = collections.Counter()
    for ln in open(f):
        ln = ln.split('//')[0].strip()
        if not ln or ln.startswith(('.','//','{','}')) or ln.endswith(':'): continue
        if ln.startswith('@'): ln = ln.split(None,1)[1] if ' ' in ln else ln
        m = re.match(r'([a-z][a-z0-9_.]*)', ln)
        if m: ops[m.group(1)] += 1
    return ops

def sass_ops(f):
    ops = collections.Counter()
    for ln in open(f):
        m = re.search(r'\*/\s+(@!?\w+\s+)?([A-Z][A-Z0-9_.]*)', ln)
        if m: ops[m.group(2)] += 1
    return ops

known_p, known_s = collections.Counter(), collections.Counter()
for k in PASSING:
    if os.path.exists(f'corpus/{k}.ptx'):
        known_p += ptx_ops(f'corpus/{k}.ptx'); known_s += sass_ops(f'corpus/{k}.sass')

want = sys.argv[1:]
for k in want:
    p, s = ptx_ops(f'corpus/{k}.ptx'), sass_ops(f'corpus/{k}.sass')
    up = {o:c for o,c in p.items() if o not in known_p}
    us = {o:c for o,c in s.items() if o not in known_s}
    print(f'\n=== {k}   ptx {sum(p.values())} insn / {len(up)} unknown ops'
          f'   sass {sum(s.values())} insn / {len(us)} unknown ops')
    print('   PTX  :', ' '.join(f'{o}x{c}' for o,c in sorted(up.items())) or '(none)')
    print('   SASS :', ' '.join(f'{o}x{c}' for o,c in sorted(us.items())) or '(none)')
