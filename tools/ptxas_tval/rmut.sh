#!/bin/bash
# Mutation table for the two RANKING measurements -- `gap.py --rank` (cost and
# reach) and `loopgap.py` (why loopval refuses each loop kernel).
#
# CONTROL ROW IS FIRST and must be read first: a table where every row moves the
# same column is reporting the state of the tree, not the mutations.  The
# RESTORED BASELINE row at the bottom is read second -- if it does not match the
# control, the archive did not restore and every row above it is contaminated.
cd "$(dirname "$0")"
./mkbase.sh rank_base.tgz || exit 1

SUB="bn254_permute ptx_subword_ops gemm_fp8_1024 bn254_fr_mul bn254_msm_bucket"

run(){   # $1 = label
  timeout 600 python3 gap.py --rank $SUB > /tmp/_rk.txt 2>&1
  st=$(grep -c 'SETUP FAILURE' /tmp/_rk.txt)
  rb=$(awk '/=== REACH/,0' /tmp/_rk.txt | awk '$2=="bra"{print $1}')
  timeout 900 python3 loopgap.py > /tmp/_lg.txt 2>&1
  nk=$(head -1 /tmp/_lg.txt | awk '{print $1}')
  mu=$(awk '/more than one back edge/{s+=$1} END{print s+0}' /tmp/_lg.txt)
  ze=$(grep -c 'found NO back edge' /tmp/_lg.txt)
  printf '%-56s  setup=%-3s reach_bra=%-4s | loopgap: %-4s kernels multi=%-4s zero=%s\n' \
         "$1" "${st:-ERR}" "${rb:-ERR}" "${nk:-ERR}" "${mu:-ERR}" "${ze:-ERR}"
}

printf '%-56s  %s\n' 'MUTATION' 'gap.py --rank (subset)          | loopgap.py (whole corpus)'
./restore.sh >/dev/null; rm -rf __pycache__; run 'R0 CONTROL: two independent inits reordered (no-op)'

# --- R0 applied: a genuine no-op, to show the columns are not order-sensitive
./restore.sh >/dev/null
python3 - <<'P'
s=open('gap.py').read()
s=s.replace("    reach = collections.Counter()\n    cost = []\n",
            "    cost = []\n    reach = collections.Counter()\n",1)
open('gap.py','w').write(s)
P
rm -rf __pycache__; run 'R0 CONTROL (applied)'

# --- R1: the loop census examines nothing.  The floor must fire; without it a
# --- census that looked at no kernel reports "no refusals" perfectly.
./restore.sh >/dev/null
sed -i 's/^    return re.search(r.\^\\s\*(@\\S+\\s+)?bra(\\.uni)?\\s., open(ptxf).read(), re.M) is not None$/    return False/' loopgap.py
grep -q 'return False' loopgap.py || echo '  !! R1 DID NOT APPLY'
rm -rf __pycache__; run 'R1 loopgap examines no kernels (floor must fire)'

# --- R2: collapse ZERO back edges into the more-than-one bucket.  Both are
# --- phrased "has N back edges" by loopval; merging them reports the larger.
./restore.sh >/dev/null
python3 - <<'P'
s=open('loopgap.py').read()
a=s.index("        return (f'{side}: loop finder found NO back edge' if n == 0")
b=s.index("    return re.sub(r'\\s+', ' ', msg)")
s=s[:a]+"        return f'{side}: has N back edges'\n"+s[b:]
open('loopgap.py','w').write(s)
P
rm -rf __pycache__; run 'R2 zero and >1 back edges folded into one bucket'

# --- R3: detect control flow the way gap.py attributes it -- unpredicated only.
# --- 42, not 37: gap.py ALSO loses 5 kernels to setup failures, so the two
# --- under-reporting causes are 6 + 5 and this probe isolates the first.
./restore.sh >/dev/null
sed -i "s|r'\^\\\\s\*(@\\\\S+\\\\s+)?bra(\\\\.uni)?\\\\s'|r'^\\\\s*bra(\\\\.uni)?\\\\s'|" loopgap.py
rm -rf __pycache__; run 'R3 control flow detected unpredicated-only (48 -> 42)'

# --- R4: a setup failure stops being flagged and reads as a cheap kernel.
./restore.sh >/dev/null
sed -i 's/^        tag = f.  \[SETUP FAILURE on.*$/        tag = ""/' gap.py
rm -rf __pycache__; run 'R4 setup failures no longer flagged in the cost ranking'

# --- R5: reach stops counting kernels and becomes presence.
./restore.sh >/dev/null
sed -i 's/^            reach\[o\] += 1$/            reach[o] = 1/' gap.py
rm -rf __pycache__; run 'R5 reach counts presence, not kernels blocked'

./restore.sh >/dev/null; rm -rf __pycache__
printf '\n'; run 'RESTORED BASELINE'
rm -f rank_base.tgz
