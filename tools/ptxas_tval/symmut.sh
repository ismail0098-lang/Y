#!/bin/bash
# Mutation table for tval.py's symmetric-cut filter (symmetric_cuts and its
# wiring into the partial-sum sweep and the store check).  CONTROL ROW FIRST,
# BASE at the top and bottom.  Two columns: `tval.py --selftest` and the whole
# of regress.sh (which runs the selftest too).  Graded on exit codes.
#
# What no row here can see, stated rather than implied: the WIRING into the
# sweep (S6) and the store check (S7).  No standing fixture reaches a cut only
# one side contains; the field kernel it was measured on takes hours per run.
#
# Every step names its target LITERALLY in open('...') so mutgate.py can see it.
cd "$(dirname "$0")"
./mkbase.sh guard_base.tgz || exit 1
row(){   # $1 = label
  touch *.py
  st=$(timeout 120 python3 tval.py --selftest 2>&1); src=$?
  f=$(printf '%s\n' "$st" | grep -m1 '^FAIL' | cut -c1-70)
  [ -z "$f" ] && f=$(printf '%s\n' "$st" | tail -1 | cut -c1-70)
  rg=$(bash regress.sh 2>&1); rrc=$?
  printf '%-56s selftest rc=%-3s regress rc=%-3s %s\n' "$1" "$src" "$rrc" "$f"
}
./restore.sh >/dev/null; row "BASE"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tval.py').read(); a="    iS, iP = _ids(rawS), _ids(rawP)\n"
assert s.count(a)==1; open('tval.py','w').write(s.replace(a,"    iP = _ids(rawP)\n    iS = _ids(rawS)\n"))
P
row "C0 CONTROL: the two id sets computed in the other order"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tval.py').read(); a="            if (cutsS[k][0].get_id() in iS) == (cutsP[k][0].get_id() in iP)]\n"
assert s.count(a)==1; open('tval.py','w').write(s.replace(a,"            if True]\n"))
P
row "S1 drop nothing (every cut kept: the state before)"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tval.py').read(); a="            if (cutsS[k][0].get_id() in iS) == (cutsP[k][0].get_id() in iP)]\n"
assert s.count(a)==1; open('tval.py','w').write(s.replace(a,"            if False]\n"))
P
row "S2 OVER-DROP: every cut dropped"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tval.py').read(); a="            if (cutsS[k][0].get_id() in iS) == (cutsP[k][0].get_id() in iP)]\n"
assert s.count(a)==1; open('tval.py','w').write(s.replace(a,"            if (cutsS[k][0].get_id() in iS) != (cutsP[k][0].get_id() in iP)]\n"))
P
row "S3 wrong direction: keep exactly the asymmetric cuts"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tval.py').read(); a="            if (cutsS[k][0].get_id() in iS) == (cutsP[k][0].get_id() in iP)]\n"
assert s.count(a)==1; open('tval.py','w').write(s.replace(a,"            if (cutsS[k][0].get_id() in iS)]\n"))
P
row "S4 only the S side consulted"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tval.py').read(); a="    return bad\n\ndef build("
assert s.count(a)==1; s=s.replace(a,"    return []\n\ndef build(")
a="            if (cutsS[k][0].get_id() in iS) == (cutsP[k][0].get_id() in iP)]\n"
assert s.count(a)==1; open('tval.py','w').write(s.replace(a,"            if True]\n"))
P
row "S5 S1 with the selftest neutered (compound)"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tval.py').read(); a="                cS, cP, _ = symmetric_cuts(subD[0], subD[1], Sd.wide[j][2], Pd.wide[i][2])\n"
assert s.count(a)==1; open('tval.py','w').write(s.replace(a,"                cS, cP, _ = subD[0], subD[1], 0\n"))
P
row "S6 the direct posing given every cut (sweep wiring)"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tval.py').read(); a="            if ndw or ndd:\n"
assert s.count(a)==1; open('tval.py','w').write(s.replace(a,"            if False:\n"))
P
row "S7 the store retry never runs (store wiring)"

./restore.sh >/dev/null; row "BASE (bottom)"
rm -f guard_base.tgz
