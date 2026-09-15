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
# Baseline: UUVUUVVVU RR RVURRRR UUU   (RRRUUVVVU RR RVURRRR RRU before the
# store-ordered model: las_sass/las_ptx refuted rather than refused, lsls validated)
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

# RETIRED Z1: it mutated the read-back REFUSAL or the single store width, both replaced by the
# store-ordered model, so its anchor no longer exists and it could only print
# PATCH DID NOT APPLY.  Successor: rtmut.sh R1/R1b.

# RETIRED Z2: it mutated the read-back REFUSAL or the single store width, both replaced by the
# store-ordered model, so its anchor no longer exists and it could only print
# PATCH DID NOT APPLY.  Successor: rtmut.sh R1b.

# RETIRED Z3: it mutated the read-back REFUSAL or the single store width, both replaced by the
# store-ordered model, so its anchor no longer exists and it could only print
# PATCH DID NOT APPLY.  Successor: rtmut.sh R1b.

M <<'P'
s=open('tval.py').read()
a="        reord = memorder.reorder_obligations(list(P0.stores), sperm)\n"
assert s.count(a)==1
open('tval.py','w').write(s.replace(a,"        reord = []\n"))
P
run 'Z4 tval: reorder obligation removed'

# RETIRED Z5: it mutated the read-back REFUSAL or the single store width, both replaced by the
# store-ordered model, so its anchor no longer exists and it could only print
# PATCH DID NOT APPLY.  Successor: rtmut.sh R1b (the batch rows).

M <<'P'
s=open('batch.py').read()
a="        reord = memorder.reorder_obligations(list(P.stores), sperm)\n"
assert s.count(a)==1
open('batch.py','w').write(s.replace(a,"        reord = []\n"))
P
run 'Z6 batch: reorder obligation removed'

M <<'P'
s=open('loopval.py').read()
a="""    memorder.require_no_read_back_across('PTX', [pp, pg0, pb0, pg0, pb0, pg0, pe0])
    memorder.require_no_read_back_across('SASS', [sprol, sb0, sb0, se0])
"""
assert s.count(a)==1
open('loopval.py','w').write(s.replace(a,""))
P
run 'Z7 loopval: read-back check removed'

M <<'P'
s=open('loopval.py').read()
a="""    memorder.require_no_read_back_across('PTX', [pp, pg0, pb0, pg0, pb0, pg0, pe0])
    memorder.require_no_read_back_across('SASS', [sprol, sb0, sb0, se0])
"""
b="""    memorder.require_no_read_back_across('PTX', [pp, pg0, pb0, pg0, pe0])
    memorder.require_no_read_back_across('SASS', [sprol, sb0, se0])
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

# RETIRED Z15: it mutated the read-back REFUSAL or the single store width, both replaced by the
# store-ordered model, so its anchor no longer exists and it could only print
# PATCH DID NOT APPLY.  Successor: memorder's private self-check (mixed widths).

# RETIRED Z15b: it mutated the read-back REFUSAL or the single store width, both replaced by the
# store-ordered model, so its anchor no longer exists and it could only print
# PATCH DID NOT APPLY.  Successor: memorder's private self-check (mixed widths).

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

# RETIRED Z17: it mutated the read-back REFUSAL or the single store width, both replaced by the
# store-ordered model, so its anchor no longer exists and it could only print
# PATCH DID NOT APPLY.  Successor: rtmut.sh R16.

./restore.sh >/dev/null; rm -rf __pycache__; touch *.py; run 'BASE (bottom, restored)'
rm -f mem_base.tgz guard_base.tgz
touch "$OUT/DONE"
