#!/bin/bash
# Mutation table for the EFFECT-MODEL work: memorder.py (read-back refusal,
# reorder obligation), loopval's prologue-store / can-end / stores-nothing
# refusals, and ptxexec's `ret`.
#
# CONTROL ROW IS FIRST and is read first; the RESTORED BASELINE row at the
# bottom is read second.  A table whose rows all move the same column is
# reporting the state of the tree, not the mutations.
#
# `import` is its own column because memorder.py self-checks AT IMPORT, so a
# mutation of a helper fails every tool at once -- a different diagnosis from a
# fixture moving.  The COMPOUND rows (`b`) remove that self-check as well, which
# is what shows the behavioural fixtures reach the helper too.
#
# The `mem` column is one letter per new row, in this order, so a row reports
# WHICH results moved rather than only that regress failed:
#   straight  las_sass las_ptx lsls swap_alias swap_off3 swap_off4 off3 pret pret_wrong
#   pstore    pstore pstore_wrong
#   loop      loop_ls loop_swap loop_swap_wrong loop_swap_exit loop_body_exit loop_nostore loop_ret_wrong
#   batch     smemval:las_sass smemval:las_ptx smemval:swap_alias
# Baseline: RRRUUVVVU RR RVURRRR RRU
cd "$(dirname "$0")"
./mkbase.sh mem_base.tgz || exit 1
OUT="${MEMMUT_OUT:-/tmp/_memmut}"; mkdir -p "$OUT"

run(){   # $1 = label
  if python3 -c 'import memorder, ptxexec, sassexec, tval, batch, loopval, smemval' >/dev/null 2>&1
  then imp=ok; else imp=FAIL; fi
  timeout 1800 ./regress.sh >"$OUT/r.txt" 2>&1; r=$?
  mem=$(python3 - "$OUT/r.txt" <<'PY'
import sys
t = open(sys.argv[1]).read().splitlines()
def v(label):
    for l in t:
        if l.startswith(label + ' ') or l.startswith(label + '\t'):
            w = l[len(label):].split()
            return {'VALIDATED': 'V', 'UNPROVED': 'U', 'REFUSED': 'R'}.get(w[0] if w else '', '?')
    return '?'
g = lambda ns, p='mem/': ''.join(v(p + n) for n in ns)
print(g('las_sass las_ptx lsls swap_alias swap_off3 swap_off4 off3 pret pret_wrong'.split()),
      g('pstore pstore_wrong'.split()),
      g('loop_ls loop_swap loop_swap_wrong loop_swap_exit loop_body_exit loop_nostore loop_ret_wrong'.split()),
      g('las_sass las_ptx swap_alias'.split(), 'smemval mem/'))
PY
)
  std=$(grep -cE '^(fma|neg|max|bn254|ptx_carry|exact_pv|naive_gemm|smem_round)[a-z_0-9]* +(VALIDATED|UNPROVED)' "$OUT/r.txt")
  timeout 600 python3 liftgap.py --selftest >"$OUT/l.txt" 2>&1; l=$?
  timeout 600 python3 frontier.py --selftest >"$OUT/f.txt" 2>&1; f=$?
  printf '%-62s import=%-4s regress=%-4s std=%2s mem=[%s] liftgap=%-4s frontier=%s\n' \
    "$1" "$imp" "$([ $r = 0 ] && echo ok || echo FAIL)" "$std" "$mem" \
    "$([ $l = 0 ] && echo ok || echo FAIL)" "$([ $f = 0 ] && echo ok || echo FAIL)" | tee -a "$OUT/progress"
}

# Restore, apply the patch on stdin, kill bytecode, and TOUCH so nothing
# stale is served (restore.sh already removes __pycache__).
M(){ ./restore.sh >/dev/null; python3 - || echo "PATCH DID NOT APPLY" | tee -a "$OUT/progress"; rm -rf __pycache__; touch *.py; }

: > "$OUT/progress"
./restore.sh >/dev/null; rm -rf __pycache__; run 'BASE (top)'

M <<'P'
s=open('memorder.py').read()
a="    extend = insert = pop = remove = clear = sort = reverse = _grow_only\n"
b="    __iadd__ = __setitem__ = __delitem__ = _grow_only\n"
assert s.count(a+b)==1
open('memorder.py','w').write(s.replace(a+b, b+a))
P
run 'Z0 CONTROL: two independent assignments reordered (no-op)'

M <<'P'
s=open('tval.py').read()
a="""        memorder.require_no_read_back('PTX', [P0])
        memorder.require_no_read_back('SASS', [S0])
"""
assert s.count(a)==1
open('tval.py','w').write(s.replace(a,"        pass\n"))
P
run 'Z1 tval: read-back check removed (the original state)'

M <<'P'
s=open('tval.py').read()
a="        memorder.require_no_read_back('SASS', [S0])\n"
assert s.count(a)==1
open('tval.py','w').write(s.replace(a,""))
P
run 'Z2 tval: PTX side only'

M <<'P'
s=open('tval.py').read()
a="        memorder.require_no_read_back('PTX', [P0])\n"
assert s.count(a)==1
open('tval.py','w').write(s.replace(a,""))
P
run 'Z3 tval: SASS side only'

M <<'P'
s=open('tval.py').read()
a="        reord = memorder.reorder_obligations(list(P0.stores), sperm)\n"
assert s.count(a)==1
open('tval.py','w').write(s.replace(a,"        reord = []\n"))
P
run 'Z4 tval: reorder obligation removed'

M <<'P'
s=open('batch.py').read()
a="""        memorder.require_no_read_back('PTX', [P])
        memorder.require_no_read_back('SASS', [S])
"""
assert s.count(a)==1
open('batch.py','w').write(s.replace(a,"        pass\n"))
P
run 'Z5 batch: read-back check removed (third validator)'

M <<'P'
s=open('batch.py').read()
a="        reord = memorder.reorder_obligations(list(P.stores), sperm)\n"
assert s.count(a)==1
open('batch.py','w').write(s.replace(a,"        reord = []\n"))
P
run 'Z6 batch: reorder obligation removed'

M <<'P'
s=open('loopval.py').read()
a="""    memorder.require_no_read_back('PTX', [pp, pg0, pb0, pg0, pb0, pg0, pe0])
    memorder.require_no_read_back('SASS', [sprol, sb0, sb0, se0])
"""
assert s.count(a)==1
open('loopval.py','w').write(s.replace(a,""))
P
run 'Z7 loopval: read-back check removed'

M <<'P'
s=open('loopval.py').read()
a="""    memorder.require_no_read_back('PTX', [pp, pg0, pb0, pg0, pb0, pg0, pe0])
    memorder.require_no_read_back('SASS', [sprol, sb0, sb0, se0])
"""
b="""    memorder.require_no_read_back('PTX', [pp, pg0, pb0, pg0, pe0])
    memorder.require_no_read_back('SASS', [sprol, sb0, se0])
"""
assert s.count(a)==1
open('loopval.py','w').write(s.replace(a,b))
P
run 'Z8 loopval: body run ONCE, not twice'

M <<'P'
s=open('loopval.py').read()
a="""        if region.stores:
            raise memorder.Refusal("""
assert s.count(a)==1
open('loopval.py','w').write(s.replace(a,"""        if False:
            raise memorder.Refusal("""))
P
run 'Z9 loopval: prologue-store refusal removed (the original state)'

M <<'P'
s=open('loopval.py').read()
a="        if not is_true(simplify(region.alive)):\n"
assert s.count(a)==1
open('loopval.py','w').write(s.replace(a,"        if False:\n"))
P
run 'Z10 loopval: can-end refusal removed (the original state)'

M <<'P'
s=open('loopval.py').read()
a="""    for where, region in (('SASS prologue', sprol), ('SASS loop body', sb0),
                          ('PTX prologue', pp), ('PTX loop header', pg0),
                          ('PTX loop body', pb0)):"""
b="""    for where, region in (('SASS prologue', sprol), ('SASS loop body', sb0)):"""
assert s.count(a)==1
open('loopval.py','w').write(s.replace(a,b))
P
run 'Z10b loopval: can-end refusal on the SASS side only'

M <<'P'
s=open('loopval.py').read()
a="    if not pe0.stores and not se0.stores:\n"
assert s.count(a)==1
open('loopval.py','w').write(s.replace(a,"    if False:\n"))
P
run 'Z11 loopval: stores-nothing refusal removed (the original state)'

M <<'P'
s=open('loopval.py').read()
a="    for (i, j), claim in memorder.reorder_obligations(list(pe.stores), perm):\n"
assert s.count(a)==1
open('loopval.py','w').write(s.replace(a,"    for (i, j), claim in []:\n"))
P
run 'Z12 loopval: reorder obligation removed'

M <<'P'
s=open('ptxexec.py').read()
a="            self.alive = simplify(And(self.alive, Not(g)))\n"
assert s.count(a)==1
open('ptxexec.py','w').write(s.replace(a,"            pass\n"))
P
run 'Z13 ptxexec: ret is a no-op again (the original state)'

M <<'P'
s=open('ptxexec.py').read()
a="            g = self.alive if is_true(g) else And(self.alive, g)\n"
assert s.count(a)==1
open('ptxexec.py','w').write(s.replace(a,"            pass\n"))
P
run 'Z14 ptxexec: alive narrowed but never applied to effects'

M <<'P'
s=open('memorder.py').read()
a="STORE_WIDTH_BYTES = 4\n"
assert s.count(a)==1
open('memorder.py','w').write(s.replace(a,"STORE_WIDTH_BYTES = 3\n"))
P
run 'Z15 memorder: store width 3 (the import self-check)'

M <<'P'
s=open('memorder.py').read()
a="STORE_WIDTH_BYTES = 4\n"
b="\n_self_check()\n"
assert s.count(a)==1 and s.count(b)==1
s=s.replace(a,"STORE_WIDTH_BYTES = 3\n").replace(b,"\n")
# the width is also the value size the shape check demands, so keep that at 32
s=s.replace("v.size() == 8 * STORE_WIDTH_BYTES","v.size() == 32")
open('memorder.py','w').write(s)
P
run 'Z15b COMPOUND: width 3 AND the self-check removed'

M <<'P'
s=open('memorder.py').read()
a="        self._order.append(self._tag)\n"
assert s.count(a)==1
open('memorder.py','w').write(s.replace(a,""))
P
run 'Z16 memorder: the trace stops recording order (import self-check)'

M <<'P'
s=open('memorder.py').read()
a="        self._order.append(self._tag)\n"
b="\n_self_check()\n"
assert s.count(a)==1 and s.count(b)==1
open('memorder.py','w').write(s.replace(a,"").replace(b,"\n"))
P
run 'Z16b COMPOUND: order not recorded AND the self-check removed'

M <<'P'
s=open('memorder.py').read()
a="    if first >= 0 and 'L' in seq[first:]:\n"
assert s.count(a)==1
open('memorder.py','w').write(s.replace(a,"    if 'L' in seq and 'S' in seq:\n"))
P
run 'Z17 OVER-REFUSAL: refuse any kernel with both a load and a store'

./restore.sh >/dev/null; rm -rf __pycache__; touch *.py; run 'BASE (bottom, restored)'
rm -f mem_base.tgz guard_base.tgz
touch "$OUT/DONE"
