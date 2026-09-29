#!/bin/bash
# Mutation table for the division-lowering referees (divlow_abi.py and its three
# probes).  CONTROL ROW IS FIRST and must be read first; BASELINE is last and
# must match it.  Needs a CUDA device.
#
# Every step names its target LITERALLY in open('...'): mutgate.py recovers a
# step's target from the text, and a first version passing it through a Python
# variable was reported EMPTY -- no rows seen, indistinguishable from a table
# that prints nothing.
cd "$(dirname "$0")"
./mkbase.sh guard_base.tgz || exit 1
run(){   # $1 = label
  out=$(timeout 900 python3 divlow_abi.py 2>&1); rc=$?
  f=$(printf '%s\n' "$out" | grep -m1 '^FAIL' | cut -c1-110)
  printf '%-58s rc=%-2s %s\n' "$1" "$rc" "${f:-(no FAIL line)}"
}
printf '%-58s %-5s %s\n' MUTATION RC 'FIRST FAIL'
./restore.sh >/dev/null; run "BASE"

./restore.sh >/dev/null
python3 - <<'P'
s=open('divlow_abi.py').read()
a="        build('newton_abi.cu', exe['newton'])\n"; b="        build('tail_abi.cu', exe['tail'])\n"
assert s.count(a+b)==1; open('divlow_abi.py','w').write(s.replace(a+b,b+a))
P
run "D0 CONTROL: two independent builds reordered"

./restore.sh >/dev/null
python3 - <<'P'
s=open('rcpwin_abi.cu').read(); a='asm("cvt.rzi.ftz.u32.f32 %0, %1;"'
assert s.count(a)==1; open('rcpwin_abi.cu','w').write(s.replace(a,'asm("cvt.rzi.u32.f32 %0, %1;"'))
P
run "D1 window probe drops .FTZ (measures another instruction)"

./restore.sh >/dev/null
python3 - <<'P'
s=open('newton_abi.cu').read(); a='unsigned e2 = e + hi;'
assert s.count(a)==1; open('newton_abi.cu','w').write(s.replace(a,'unsigned e2 = hi;'))
P
run "D2 Newton addend read as 32-bit Rc (the pair misread)"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tail_abi.cu').read(); a='bool c2 = r1 >= d;'
assert s.count(a)==1; open('tail_abi.cu','w').write(s.replace(a,'bool c2 = false;'))
P
run "D3 the second correction removed"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tail_abi.cu').read()
for a,b in (('for (int w = 0; w < 2; w++) {','for (int w = 0; w < 1; w++) {'),
            ('for (int w=0;w<2;w++) {','for (int w=0;w<1;w++) {')):
    assert s.count(a)==1; s=s.replace(a,b)
open('tail_abi.cu','w').write(s)
P
run "D4 the tail never tries e2 = I-1"

./restore.sh >/dev/null
python3 - <<'P'
a='b += 0x0ffffffeu;'
s=open('rcpwin_abi.cu').read(); assert s.count(a)==1; open('rcpwin_abi.cu','w').write(s.replace(a,'b += 0x0fffffffu;'))
s=open('newton_abi.cu').read(); assert s.count(a)==1; open('newton_abi.cu','w').write(s.replace(a,'b += 0x0fffffffu;'))
P
run "D5 the bit-pattern bias one ulp larger"

./restore.sh >/dev/null
python3 - <<'P'
s=open('divlow_abi.py').read(); a="        if a['maxdeficit(I-e2)'] != 1:"
assert s.count(a)==1; open('divlow_abi.py','w').write(s.replace(a,"        if False:"))
P
run "D6 Lemma A tightness not asserted (no-op alone)"

./restore.sh >/dev/null
python3 - <<'P'
s=open('newton_abi.cu').read(); a='unsigned e2 = e + hi;'
assert s.count(a)==1; open('newton_abi.cu','w').write(s.replace(a,'unsigned e2 = e + hi - (hi > 0u ? 1u : 0u);'))
P
run "D7 the Newton step made one unit worse"

./restore.sh >/dev/null
python3 - <<'P'
s=open('divlow_abi.py').read(); a="        if a['maxdeficit(I-e2)'] != 1:"
assert s.count(a)==1; open('divlow_abi.py','w').write(s.replace(a,"        if False:"))
s=open('newton_abi.cu').read(); a='unsigned e2 = e + hi;'
assert s.count(a)==1; open('newton_abi.cu','w').write(s.replace(a,'unsigned e2 = e + hi - (hi > 0u ? 1u : 0u);'))
P
run "D6b D6 AND D7 (the compound)"

./restore.sh >/dev/null
echo; echo "restored; verifying baseline is back:"; run "BASE"

rm -f guard_base.tgz  # a surviving archive means this run did not finish
