#!/bin/bash
# Mutation table for the float-macro-op guard.  CONTROL ROW IS FIRST and must be
# read first: a table where every row fails the same suite is reporting the
# state of the tree, not the mutations.
cd "$(dirname "$0")"
./mkbase.sh guard_base.tgz || exit 1
run(){   # $1 = label
  ok=$(python3 -c "import fpmode" 2>&1 | tail -1)
  [ -z "$ok" ] && g="guard: ok" || g="guard: FIRES [${ok:0:52}]"
  r=$(timeout 300 python3 tval.py fma/rn.ptx fma/rn.sass 12 3 15 2>&1 | tail -1 | cut -c1-32)
  p=$(timeout 300 python3 tval.py corpus/bn254_permute.ptx corpus/bn254_permute.sass 12 3 15 2>&1 | tail -1 | cut -c1-32)
  printf '%-52s %-64s %-20s %s\n' "$1" "$g" "$r" "$p"
}
printf '%-52s %-64s %-20s %s\n' MUTATION GUARD 'fma/rn' 'bn254_permute'
./restore.sh >/dev/null; run "G0 CONTROL: two table entries reordered (no-op)"

./restore.sh >/dev/null
python3 - <<'P'
s=open('fpmode.py').read()
a=s.index("    'sin.approx.f32':"); b=s.index("    'cos.approx.f32':"); c=s.index("    'ex2.approx.f32':")
s=s[:a]+s[b:c]+s[a:b]+s[c:]
open('fpmode.py','w').write(s)
P
rm -rf __pycache__; run "G0 CONTROL (applied)"

./restore.sh >/dev/null
sed -i "s/if side == 'ptx' and name in HARDWARE_PRIMITIVES and not _via_table:/if False:/" fpmode.py
rm -rf __pycache__; run "G1 PTX side may reach a MUFU directly"

./restore.sh >/dev/null
sed -i "s/^    if not validated:$/    if False:/" fpmode.py
rm -rf __pycache__; run "G2 identification used without a device probe"

./restore.sh >/dev/null
python3 - <<'P'
s=open('fpmode.py').read()
s=s.replace("    'rcp.rn.f32': 72, 'sqrt.rn.f32': 40, 'div.rn.f32': 120,",
            "    'sqrt.rn.f32': 40,")
s=s.replace("    'sin.approx.f32':   ('MUFU_SIN',      False),",
            "    'sin.approx.f32':   ('MUFU_SIN',      False),\n"
            "    'div.rn.f32':       ('MUFU_RCP',      True),\n"
            "    'rcp.rn.f32':       ('MUFU_RCP',      True),")
open('fpmode.py','w').write(s)
P
rm -rf __pycache__; run "G3 div.rn/rcp.rn reclassified as 1:1 and validated"

./restore.sh >/dev/null
sed -i "s/^_probe = factory()$/_probe = None/" fpmode.py
rm -rf __pycache__; run "G4 import-time check removed"

./restore.sh >/dev/null
sed -i "s/^        elif op in fpmode.MACRO_OPS:$/        elif False:/" ptxexec.py
rm -rf __pycache__; run "G5 ptxexec stops naming macro-ops (falls to generic)"

./restore.sh >/dev/null
echo; echo "restored; verifying baseline is back:"; run "BASELINE"

rm -f guard_base.tgz  # a surviving archive means this run did not finish
