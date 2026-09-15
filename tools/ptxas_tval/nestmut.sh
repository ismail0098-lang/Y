#!/bin/bash
# Mutation table for nestval.py.  One row per mutation; each row restores the
# baseline, applies ONE patch (asserted to apply exactly once, or the row says
# so rather than running an unmutated tree), and prints one verdict letter per
# fixture: V validated, U unproved, R refused, C crashed.
#
#   NESTMUT_ONLY=N3   run only the rows whose label contains that text
#
# Run it from a FILE, never concurrently with anything that imports these
# modules, and never while `frontier.py` is running if a row touches a module in
# its cache key (the sassexec row is labelled so).
cd "$(dirname "$0")"
rm -rf __pycache__
BASE=$(mktemp -d)
cp nestval.py sassexec.py "$BASE/"
restore() { cp "$BASE/nestval.py" "$BASE/sassexec.py" .; touch nestval.py sassexec.py; rm -rf __pycache__; }
trap 'restore; rm -rf "$BASE"' EXIT

FIX="o1/y_cpu_matmul o1/y_cpu_matmul_w1_store_stride o1/y_cpu_matmul_w2_acc_add o1/y_cpu_matmul_w3_acc_init
     o1/y_cpu_matmul_w4_top_guard o1/y_cpu_matmul_w5_inner_guard o1/y_cpu_matmul_w6_inner_backedge
     o1/y_cpu_matmul_w7_acc_init_k0 mem/nest_accum mem/nest_accum_stale o1/naive_gemm_f32_rn
     o1/naive_gemm_f32_muladd o1/exact_pv mem/loop_swap_exit"
HEAD="y w1 w2 w3 w4 w5 w6 w7 acc stale rn muladd epv exit"

verdicts() {
  local s=""
  for t in $FIX; do
    out=$(timeout 600 python3 nestval.py "$t.ptx" "$t.sass" 20 2>&1 | tail -1)
    case "$out" in
      VALIDATED*) s="$s V" ;; UNPROVED*) s="$s U" ;; REFUSED*) s="$s R" ;; *) s="$s C" ;;
    esac
  done
  echo "$s"
}

row() {
  local label="$1" file="$2" py="$3"
  if [ -n "$NESTMUT_ONLY" ] && [[ "$label" != *"$NESTMUT_ONLY"* ]]; then return; fi
  restore
  if [ -n "$py" ]; then
    if ! python3 - "$file" <<PY
import sys
f = sys.argv[1]; s = open(f).read()
$py
open(f, 'w').write(s)
PY
    then printf '%-34s MUTATION DID NOT APPLY\n' "$label"; return; fi
  fi
  printf '%-34s %s\n' "$label" "$(verdicts)"
}

sub() {  # python for one asserted substitution
  printf 'a=%s\nb=%s\nassert s.count(a)==1, "anchor"\ns=s.replace(a,b)\n' "$1" "$2"
}

printf '%-34s  %s\n' "" "$HEAD"
row "BASE"                              nestval.py ""
row "N0 CONTROL reorder a set literal"  nestval.py "$(sub "\"names = {'FADD', 'MUL64', 'MULLO', 'MULHI'}\"" "\"names = {'MULHI', 'MULLO', 'MUL64', 'FADD'}\"")"
row "N1 exit pairs not shared"          nestval.py "$(sub "\"pmap.setdefault(a, shared[b[1] if b[0] == 'R' else f'P{b[1]}'])\"" "\"pmap.setdefault(a, fresh_like(shared[b[1] if b[0] == 'R' else f'P{b[1]}'], f'unshared{uid}_{a[0]}{a[1]}'))\"")"
row "N2 step memory not fresh"          nestval.py "$(sub "\"symS = dict(sym, mem=self.mem('step')) if d['stores'] else sym\"" "\"symS = sym\"")"
row "N3 iteration stores uncompared"    nestval.py "$(sub "\"self.compare_stores(pstores, sstores, [cont], axioms, 'iteration')\"" "\"pass\"")"
row "N4 ENTRY not posed"                nestval.py "$(sub "\"r = self.prove(sass_skip == ptx_skip)\"" "\"r = 'unsat'\"")"
row "N5 LOOPCOND not posed"             nestval.py "$(sub "\"r = self.prove(sass_cont == nxt, [cont], axioms)\"" "\"r = 'unsat'\"")"
row "N6 no commutativity instances"     nestval.py "$(sub "\"so.add(comm_instances([claim] + list(extra)))\"" "\"pass\"")"
row "N7 EXIT-guard refusal removed"     nestval.py "$(sub "\"if ST['root']['gkind'] == 'exit' and (pe.stores or se.stores):\"" "\"if False:\"")"
row "N8 BASE not posed"                 nestval.py "$(sub "\"pairs = [(a, b) for a, b in pairs if self.prove(p_out(a, pp) == s_out(b, sp)) == 'unsat']\"" "\"pairs = list(pairs)\"")"
# N8 against w3 SURVIVED the first sweep: the proposer simulates from the real
# entry values, so an accumulator starting at 1 against 0 is never PROPOSED and
# BASE never sees it.  w7 is wrong only when K == 0, which random simulation
# cannot draw, so its pair IS proposed and BASE is the one thing that refutes it.
# (A compound that made the proposer accept every kind-matching pair was tried
# and was mis-aimed: it breaks the CORRECT target too, because each SASS slot is
# seeded from its first pair, so it could not tell BASE's work from its own.)
row "N9 OVER-REFUSAL no child loops"    nestval.py "$(sub "\"        spans = [(C['g'], C['e']) for C in L['children']]\"" "\"        if L['children']: refuse('probe: no children')\n        spans = [(C['g'], C['e']) for C in L['children']]\"")"
row "N10 sassexec gaddr back to dict [census key]" sassexec.py "$(sub "\"addr = u64(self.rd(f'R{ab+1}'), self.rd(f'R{ab}'))\"" "\"addr = u64(self.R[ab+1], self.R[ab])\"")"
row "BASE again"                        nestval.py ""
