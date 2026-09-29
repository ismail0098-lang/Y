#!/bin/bash
# Mutation table for the STORE-ORDERED, BYTE-FAITHFUL memory model:
# memorder.read_through / pair_by_address / require_no_read_back_across, the
# executors' single memory path (`gload`) and address hook, the sub-word store and
# u8 conversion arms, the width checks, and sassexec's LOP3.LUT refusal.
#
# CONTROL ROW IS FIRST and is read first; the RESTORED BASELINE row at the bottom
# is read second.  A table whose rows all move the same column is reporting the
# state of the tree, not the mutations.
#
# `import` is its own column because memorder and both executors self-check AT
# IMPORT, so a mutation of a pinned helper fails every tool at once -- a different
# diagnosis from a fixture moving.  The COMPOUND rows (`b`) remove that
# self-check as well, which is what shows whether a behavioural row reaches the
# helper on its own.
#
# The `mem` column is one letter per row of interest, in this order:
#   straight  las_sass las_ptx lsls swap_off3 swap_off4 sls_alias
#   subword   corpus/ptx_subword_ops mem/subword_hoist mem/subword_widen
#   loop      loop_ls
#   batch     smemval:las_sass smemval:las_ptx smemval:lsls
# `lsls` prints its obligation count too: 10 is the address hook working.
# `abi` runs the device referee only on the rows that mutate it.
cd "$(dirname "$0")"
./mkbase.sh rt_base.tgz || exit 1
OUT="${RTMUT_OUT:-/tmp/_rtmut}"; mkdir -p "$OUT"

run(){   # $1 = label, $2 = "abi" to run the device referee
  if [ "$SKIP" = 1 ]; then SKIP=; return; fi
  if python3 -c 'import memorder, ptxexec, sassexec, tval, batch, loopval, smemval' >/dev/null 2>&1
  then imp=ok; else imp=FAIL; fi
  timeout 2400 ./regress.sh >"$OUT/r.txt" 2>&1; r=$?
  mem=$(python3 - "$OUT/r.txt" <<'PY'
import sys
t = open(sys.argv[1]).read().splitlines()
def v(label):
    for l in t:
        if l.startswith(label + ' '):
            w = l[len(label):].split()
            return {'VALIDATED': 'V', 'UNPROVED': 'U', 'REFUSED': 'R'}.get(w[0] if w else '', '?')
    return '?'
def n(label):
    for l in t:
        if l.startswith(label + ' '):
            w = l[len(label):].split()
            return w[1] if len(w) > 1 else '?'
    return '?'
g = lambda ns, p='mem/': ''.join(v(p + x) for x in ns)
print(g('las_sass las_ptx lsls swap_off3 swap_off4 sls_alias'.split()) + f'({n("mem/lsls")})',
      g('corpus/ptx_subword_ops mem/subword_hoist mem/subword_widen'.split(), ''),
      g(['loop_ls']),
      g('las_sass las_ptx lsls'.split(), 'smemval mem/'))
PY
)
  timeout 600 python3 liftgap.py --selftest >"$OUT/l.txt" 2>&1; l=$?
  abi=-
  if [ "$2" = abi ]; then
    timeout 900 python3 subword_abi.py >"$OUT/a.txt" 2>&1 && abi=ok || abi=FAIL
  fi
  printf '%-64s import=%-4s regress=%-4s mem=[%s] liftgap=%-4s abi=%s\n' \
    "$1" "$imp" "$([ $r = 0 ] && echo ok || echo FAIL)" "$mem" \
    "$([ $l = 0 ] && echo ok || echo FAIL)" "$abi" | tee -a "$OUT/progress"
}

# Restore, apply the patch on stdin, kill bytecode, and TOUCH so nothing stale is
# served.  A patch whose anchor is absent says so rather than running unmutated.
# RTMUT_ONLY: an ERE over row labels; a row it does not match neither patches nor
# runs (its stdin is drained).  BASE rows always run, so a filtered run still
# reads its control and its restored baseline.
SKIP=
want(){ [ -z "$RTMUT_ONLY" ] || printf '%s\n' "$1" | grep -qE "$RTMUT_ONLY"; }
M(){ if ! want "$1"; then cat >/dev/null; SKIP=1; return; fi
     SKIP=; ./restore.sh >/dev/null; python3 - || echo "PATCH DID NOT APPLY" | tee -a "$OUT/progress"; rm -rf __pycache__; touch *.py; }

: > "$OUT/progress"
./restore.sh >/dev/null; rm -rf __pycache__; run 'BASE (top)' abi

M "R0 CONTROL: two self-check cases reordered (no-op)" <<'P'
s=open('memorder.py').read()
a="        ([(A + lit(5, 64), lit(0x11223344, 32), T)], 4, 0x44332211),       # disjoint\n"
b="        ([(A + lit(2, 64), lit(0x7766, 16), T)], 1, 0x44332211),           # misses byte 0\n"
assert s.count(b+a)==1
open('memorder.py','w').write(s.replace(b+a, a+b))
P
run 'R0 CONTROL: two self-check cases reordered (no-op)'

M "R1 read_through ignores the stores (the original model)" <<'P'
s=open('memorder.py').read()
a="    if not stores:\n        return base\n    if not (is_bv(addr)"
assert s.count(a)==1
open('memorder.py','w').write(s.replace(a,"    return base\n    if not (is_bv(addr)"))
P
run 'R1 read_through ignores the stores (the original model)'

M "R1b COMPOUND: store-blind read AND the private self-check removed" <<'P'
s=open('memorder.py').read()
a="    if not stores:\n        return base\n    if not (is_bv(addr)"
b="    _private_self_check()\n"
assert s.count(a)==1 and s.count(b)==1
s=s.replace(a,"    return base\n    if not (is_bv(addr)").replace(b,"")
open('memorder.py','w').write(s)
P
run 'R1b COMPOUND: store-blind read AND the private self-check removed'

M "R2 an EARLIER store wins (fold order reversed)" <<'P'
s=open('memorder.py').read()
a="            for (s, v, g), w in zip(stores, widths):\n"
assert s.count(a)==1
open('memorder.py','w').write(s.replace(a,"            for (s, v, g), w in reversed(list(zip(stores, widths))):\n"))
P
run 'R2 an EARLIER store wins (fold order reversed)'

M "R3 store bytes read BIG-endian" <<'P'
s=open('memorder.py').read()
a="        r = If(d == BitVecVal(j, 64, d.ctx), Extract(8 * j + 7, 8 * j, v), r)\n"
assert s.count(a)==1
w="        r = If(d == BitVecVal(j, 64, d.ctx), Extract(v.size() - 8 * j - 1, v.size() - 8 * j - 8, v), r)\n"
s=s.replace(a,w).replace("    r = Extract(7, 0, v)\n","    r = Extract(v.size() - 1, v.size() - 8, v)\n")
open('memorder.py','w').write(s)
P
run 'R3 store bytes read BIG-endian'

M "R4 ld.global.f32 bypasses gload (reads the initial array)" <<'P'
s=open('ptxexec.py').read()
a="            self.wf(ops[0], self.gload(addr, i, 0), g)\n"
assert s.count(a)==1
open('ptxexec.py','w').write(s.replace(a,"            self.wf(ops[0], self.sym['abstract'](i,0) if 'abstract' in self.sym else Select(self.sym['mem'], addr), g)\n"))
P
run 'R4 ld.global.f32 bypasses gload (reads the initial array)'

M "R4b COMPOUND: the bypass AND the import pin removed" <<'P'
s=open('ptxexec.py').read()
a="            self.wf(ops[0], self.gload(addr, i, 0), g)\n"
b="memorder.pin_one_memory_path(Ptx)\n"
assert s.count(a)==1 and s.count(b)==1
s=s.replace(a,"            self.wf(ops[0], self.sym['abstract'](i,0) if 'abstract' in self.sym else Select(self.sym['mem'], addr), g)\n").replace(b,"")
open('ptxexec.py','w').write(s)
P
run 'R4b COMPOUND: the bypass AND the import pin removed'

M "R5 tval: store width check removed" <<'P'
s=open('tval.py').read()
a="        if wp != ws: return 'UNPROVED', f'store {i} width: ptx {wp} bits, sass {ws} bits', nobl\n"
assert s.count(a)==1
open('tval.py','w').write(s.replace(a,""))
P
run 'R5 tval: store width check removed'

M "R6 tval: the address hook removed (a performance property)" <<'P'
s=open('tval.py').read()
a="    symS['abstract_addr'] = lambda kind, j:"
assert s.count(a)==1
open('tval.py','w').write(s.replace(a,"    _unused = lambda kind, j:"))
P
run 'R6 tval: the address hook removed (a performance property)'

M "R7 pair_by_address: equal-address groups refused again" <<'P'
s=open('memorder.py').read()
a="        if not hs or len(hs) != len(members) or hs & claimed:\n"
assert s.count(a)==1
open('memorder.py','w').write(s.replace(a,"        if len(hs) != 1 or len(members) != 1 or hs & claimed:\n"))
P
run 'R7 pair_by_address: equal-address groups refused again'

M "R8 pair_by_address: a group paired in REVERSE occurrence order" <<'P'
s=open('memorder.py').read()
a="        for m, j in zip(members, sorted(hs)):\n"
assert s.count(a)==1
open('memorder.py','w').write(s.replace(a,"        for m, j in zip(members, sorted(hs, reverse=True)):\n"))
P
run 'R8 pair_by_address: a group paired in REVERSE occurrence order'

M "R9 loopval: cross-region read-back refusal removed" <<'P'
s=open('loopval.py').read()
a="""    memorder.require_no_read_back_across('PTX', [pp, pg0, pb0, pg0, pb0, pg0, pe0])
    memorder.require_no_read_back_across('SASS', [sprol, sb0, sb0, se0])
"""
assert s.count(a)==1
open('loopval.py','w').write(s.replace(a,""))
P
run 'R9 loopval: cross-region read-back refusal removed'

M "R10 loopval: body run ONCE, not twice" <<'P'
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
run 'R10 loopval: body run ONCE, not twice'

M "R11 ptxexec: a sub-word store records the whole word" <<'P'
s=open('ptxexec.py').read()
a="            self.stores.append((addr, Extract(nb - 1, 0, self.R(ops[1])), g))\n"
assert s.count(a)==1
open('ptxexec.py','w').write(s.replace(a,"            self.stores.append((addr, self.R(ops[1]), g))\n"))
P
run 'R11 ptxexec: a sub-word store records the whole word'

M "R12 ptxexec: cvt.u8.u32 modelled as SATURATION" <<'P'
s=open('ptxexec.py').read()
a="            self.wr(ops[0], ZeroExt(24, Extract(7, 0, self.R(ops[1]))), g)\n"
assert s.count(a)==1
open('ptxexec.py','w').write(s.replace(a,"            self.wr(ops[0], If(ULE(self.R(ops[1]), bv(255)), self.R(ops[1]), bv(255)), g)\n"))
P
run 'R12 ptxexec: cvt.u8.u32 modelled as SATURATION'

M "R13 referee: expects saturation" <<'P'
s=open('subword_abi.py').read()
a="        elif o0 != (v & 0xFF):\n"
assert s.count(a)==1
open('subword_abi.py','w').write(s.replace(a,"        elif o0 != min(v, 0xFF):\n"))
P
run 'R13 referee: expects saturation' abi

M "R14 referee: asserts every access runs (no fault expected)" <<'P'
s=open('subword_abi.py').read()
a="    bad = [k for k, v in rows.items() if (v == 'ok') != (k[2] % k[1] == 0)]\n"
assert s.count(a)==1
open('subword_abi.py','w').write(s.replace(a,"    bad = [k for k, v in rows.items() if (v == 'ok') != True]\n"))
P
run 'R14 referee: asserts every access runs (no fault expected)' abi

M "R15 sassexec: LOP3.LUT sixth operand no longer checked" <<'P'
s=open('sassexec.py').read()
a="            if len(ops) != 6 or ops[5] != '!PT':\n"
assert s.count(a)==1
open('sassexec.py','w').write(s.replace(a,"            if False:\n"))
P
run 'R15 sassexec: LOP3.LUT sixth operand no longer checked'

M "R16 OVER-REFUSAL: every read-back refused again" <<'P'
s=open('memorder.py').read()
a="    if not stores:\n        return base\n"
assert s.count(a)==1
open('memorder.py','w').write(s.replace(a,"    if not stores:\n        return base\n    raise Refusal('memorder: over-refusal probe: every read-back refused')\n"))
P
run 'R16 OVER-REFUSAL: every read-back refused again'

M "R15b COMPOUND: LOP3.LUT operand check AND its import pin removed" <<'P'
s=open('sassexec.py').read()
a="            if len(ops) != 6 or ops[5] != '!PT':\n"
b=("        st.step('LOP3.LUT R0, R1, R2, R3, 0xc0, P1', 0x10)\n"
   "        raise AssertionError('sassexec: a LOP3.LUT with a non-`!PT` sixth operand was '\n"
   "                             'read as if it were absent  (guessing, not refusing)')\n")
assert s.count(a)==1 and s.count(b)==1
s=s.replace(a,"            if False:\n").replace(b,"        pass\n")
open('sassexec.py','w').write(s)
P
run 'R15b COMPOUND: LOP3.LUT operand check AND its import pin removed'

M "R16b COMPOUND: every read-back refused AND the private self-check removed" <<'P'
s=open('memorder.py').read()
a="    if not stores:\n        return base\n"
b="    _private_self_check()\n"
assert s.count(a)==1 and s.count(b)==1
s=s.replace(a,"    if not stores:\n        return base\n    raise Refusal('memorder: over-refusal probe: every read-back refused')\n").replace(b,"")
open('memorder.py','w').write(s)
P
run 'R16b COMPOUND: every read-back refused AND the private self-check removed'

./restore.sh >/dev/null; rm -rf __pycache__; touch *.py; run 'BASE (bottom, restored)' abi
rm -f rt_base.tgz guard_base.tgz
touch "$OUT/DONE"
