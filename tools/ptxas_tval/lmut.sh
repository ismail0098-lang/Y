#!/bin/bash
# Mutation table for the LOOP validator.  CONTROL FIRST.
cd "$(dirname "$0")"
./mkbase.sh loop_base.tgz || exit 1
run(){ r=$(timeout 400 python3 loopval.py o1/exact_pv.ptx o1/exact_pv.sass 60 wide 2>&1 | tail -1 | cut -c1-76)
       printf '%-56s %s\n' "$1" "$r"; }
restore(){ tar xzf loop_base.tgz; rm -rf __pycache__; }
printf '%-56s %s\n' MUTATION 'exact_pv @ -O1'
restore; run "L0 CONTROL: two seed dict entries reordered (no-op)"
restore; python3 - <<'P'
s=open('loopval.py').read()
s=s.replace("        for a, b in pairs:\n            if b[0] != 'R' or b[1] in used: continue",
            "        for a, b in reversed(pairs):\n            if b[0] != 'R' or b[1] in used: continue")
open('loopval.py','w').write(s)
P
rm -rf __pycache__; run "L0b CONTROL: seed the pairs in the other order"
restore; sed -i "s/^    axioms = \[\]$/    axioms = []  # M/" loopval.py
sed -i "s/        axioms = mac64.instances(_region_exprs(pb), sym\['mul'\])/        axioms = []/" loopval.py
sed -i "s/    axioms = mac64.instances(_region_exprs(pb) + _region_exprs(pe), sym\['mul'\])/    axioms = []/" loopval.py
rm -rf __pycache__; run "L1 multiplier identity withheld"
restore; python3 - <<'P'
s=open('loopval.py').read()
s=s.replace("        sb = sassexec.run_insns(S['body'], sym, 'sass body', ss)\n        axioms",
            "        sb = sassexec.run_insns(S['body'], sym, 'sass body')\n        axioms")
open('loopval.py','w').write(s)
P
rm -rf __pycache__; run "L2 SASS body not seeded (rewrite-after instead)"
restore; python3 - <<'P'
s=open('loopval.py').read()
s=s.replace("        pb = ptxexec.run_lines(P['body'], sym, ps)\n        sb =",
            "        pb = ptxexec.run_lines(P['body'], sym)\n        sb =")
open('loopval.py','w').write(s)
P
rm -rf __pycache__; run "L3 PTX body not seeded (invariants left symbolic)"
restore; sed -i "s/    r = prove(sass_cont == substitute(cont, \*step_sub), \[cont\]); n += 1/    r = prove(sass_cont == cont, [cont]); n += 1/" loopval.py
rm -rf __pycache__; run "L4 LOOPCOND compares the guard at k, not k+1"
restore; python3 - <<'P'
s=open('loopval.py').read()
s=s.replace("        if not drop: break","        break  # no fixpoint")
open('loopval.py','w').write(s)
P
rm -rf __pycache__; run "L5 STEP does not iterate to a fixpoint"
restore; sed -i "s/^            r = prove(p_out(a, pb) == s_out(b, sb), \[ptx_cont_of(P, sym, ps)\]); n += 1$/            r = 'unsat'/" loopval.py
rm -rf __pycache__; run "L6 STEP always passes (over-accept control)"
restore; python3 - <<'P'
s=open('loopval.py').read()
s=s.replace("    for it in range(12):", "    for it in range(1):")
open('loopval.py','w').write(s)
P
rm -rf __pycache__; run "L7 fixpoint capped at one round"
restore; echo; echo "restored:"; run "BASELINE"

rm -f loop_base.tgz   # a surviving archive means this run did not finish
