#!/bin/bash
# Mutation table for intwall.py -- the per-pair bitvector-vs-Int comparison on
# tval's partial-sum obligations.  CONTROL ROW FIRST, BASE at the top and bottom.
# Three pairs at 5 s is enough to exercise every guard; the published figure is
# the 40-pair run in the doc.  Graded on the exit code and the first FAIL line.
#
# Every step names its target LITERALLY in open('...') so mutgate.py can see it.
cd "$(dirname "$0")"
./mkbase.sh guard_base.tgz || exit 1
row(){   # $1 = label
  touch *.py
  o=$(timeout 300 python3 intwall.py corpus/bn254_fr_mul_fast 3 5 2>&1); rc=$?
  f=$(printf '%s\n' "$o" | grep -m1 '^FAIL' | cut -c1-90)
  [ -z "$f" ] && f=$(printf '%s\n' "$o" | tail -1 | cut -c1-90)
  printf '%-58s rc=%-3s %s\n' "$1" "$rc" "$f"
}
./restore.sh >/dev/null; row "BASE"

./restore.sh >/dev/null
python3 - <<'P'
s=open('intwall.py').read(); a="        calls['n'] += 1\n        calls['unsat'] += r == 'unsat'\n"
assert s.count(a)==1; open('intwall.py','w').write(s.replace(a,"        calls['unsat'] += r == 'unsat'\n        calls['n'] += 1\n"))
P
row "C0 CONTROL: two independent counter updates reordered"

./restore.sh >/dev/null
python3 - <<'P'
s=open('intwall.py').read(); a="        ri = ask_int(Sd.wide[j][2], Pd.wide[i][2], __B__)\n"
assert s.count(a)==1; open('intwall.py','w').write(s.replace(a,"        ri = ask(Sd.wide[j][2], Pd.wide[i][2], __B__)\n"))
P
row "M1 the Int query is the bitvector query"

./restore.sh >/dev/null
python3 - <<'P'
s=open('intwall.py').read(); a="        r = real(*a, **k)\n"
assert s.count(a)==1; open('intwall.py','w').write(s.replace(a,"        r = 'unknown'\n"))
P
row "M2 the Int engine answers unknown to everything"

./restore.sh >/dev/null
python3 - <<'P'
s=open('intwall.py').read(); a="        ri = ask_int(Sd.wide[j][2], Pd.wide[i][2], __B__)\n"
assert s.count(a)==1; s=s.replace(a,"        ri = ask(Sd.wide[j][2], Pd.wide[i][2], __B__)\n")
a="    if calls['n'] != len(rows):\n"
assert s.count(a)==1; open('intwall.py','w').write(s.replace(a,"    if False:\n"))
P
row "M1b M1 with the query-count check removed (compound)"

./restore.sh >/dev/null
python3 - <<'P'
s=open('intwall.py').read(); a="        r = real(*a, **k)\n"
assert s.count(a)==1; s=s.replace(a,"        r = 'unknown'\n")
a="    if not calls['unsat']:\n"
assert s.count(a)==1; open('intwall.py','w').write(s.replace(a,"    if False:\n"))
P
row "M2b M2 with the proved-something check removed (compound)"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tval.py').read(); a="    todo = list(prop); rnd_pass = 0\n"
assert s.count(a)==1; open('tval.py','w').write(s.replace(a,"    todo = list(prop);  rnd_pass = 0\n"))
P
row "M3 tval's anchor moved (must refuse, not run tval unpatched)"

./restore.sh >/dev/null
python3 - <<'P'
s=open('intwall.py').read(); a="            rows.append((int(mm[1]), int(mm[2]), mm[3], mm[4]))\n"
assert s.count(a)==1; open('intwall.py','w').write(s.replace(a,"            pass\n"))
P
row "M4 the pair rows are never recorded"

./restore.sh >/dev/null
echo; echo "restored; verifying baseline is back:"; row "BASE"

rm -f guard_base.tgz  # a surviving archive means this run did not finish
