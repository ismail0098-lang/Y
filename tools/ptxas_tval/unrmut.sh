#!/bin/bash
# Mutation table for the FOURTH LAYER -- the unroll proxy and its crossing into
# the frontier.
#
# One row per mutation.  Each row restores the baseline, applies ONE patch
# (asserted to apply exactly once, or the row says so rather than running an
# unmutated tree), and prints one letter per check: . pass, F fail.
#
#   UNRMUT_ONLY=U3   run only the rows whose label contains that text
#
# READ THE CONTROL ROW FIRST and the closing BASE row second: a table whose
# control is red reports the state of the tree, not the mutation.
#
# THE `docgate` COLUMN IGNORES THE STAMP-TREE FAILURE, for the reason
# `sumut.sh` records: `frontier.py`, `liftgap.py` and `unroll.py` are all in
# `frontier.MODELS`, so ANY edit to one invalidates the stamp by construction.
# That is the stamp working, and it would make the column red on the CONTROL
# row, where it distinguishes nothing.  Every other `docgate` failure is graded.
#
# `frontier --selftest` is the column that sees the layer being CROSSED without
# paying for the minutes-long census; it runs over a three-kernel sub-corpus
# that includes `exact_pv` deliberately, because that is the kernel the layer
# blocks and without it the crossing control asserts nothing.
#
# Run it from a FILE, never concurrently with `frontier.py`.
cd "$(dirname "$0")"
rm -rf __pycache__
BASE=$(mktemp -d)
FILES="unroll.py frontier.py liftgap.py docgate.py"
for f in $FILES; do cp "$f" "$BASE/"; done
cp ../../docs/ptxas_translation_validation.md "$BASE/tvaldoc.md"
cp ../../README.md "$BASE/README.md"
restore() {
  for f in $FILES; do cp "$BASE/$f" .; touch "$f"; done
  cp "$BASE/tvaldoc.md" ../../docs/ptxas_translation_validation.md
  cp "$BASE/README.md" ../../README.md
  rm -rf __pycache__
}
trap 'restore; rm -rf "$BASE"' EXIT

checks() {
  local s=""
  timeout 600 python3 unroll.py   --selftest >/dev/null 2>&1 && s="$s." || s="${s}F"
  timeout 900 python3 frontier.py --selftest >/dev/null 2>&1 && s="$s." || s="${s}F"
  timeout 600 python3 liftgap.py  --selftest >/dev/null 2>&1 && s="$s." || s="${s}F"
  local out
  out=$(timeout 1800 python3 docgate.py 2>&1)
  if printf '%s\n' "$out" | sed '/--- positive controls/,$d' \
       | grep '^FAIL:' | grep -qv 'stamped on a different tree'
  then s="${s}F"; else s="$s."; fi
  echo "$s"
}

row() {
  local label="$1" file="$2" py="$3"
  if [ -n "$UNRMUT_ONLY" ] && [[ "$label" != *"$UNRMUT_ONLY"* ]]; then return; fi
  restore
  if [ -n "$py" ]; then
    if ! python3 - "$file" <<PY
import sys
f = sys.argv[1]; s = open(f).read()
$py
open(f, 'w').write(s)
PY
    then printf '%-50s MUTATION DID NOT APPLY\n' "$label"; return; fi
  fi
  printf '%-50s %s\n' "$label" "$(checks)"
}

sub() { printf 'a=%s\nb=%s\nassert s.count(a)==1, "anchor"\ns=s.replace(a,b)\n' "$1" "$2"; }

printf '%-50s %s\n' '' 'unroll / frontier / liftgap / docgate'
row "BASE" '' ''
row "U0 CONTROL reorder two independent patterns" unroll.py "$(sub "\"PTX_OBS = re.compile\"" "\"_UNUSED_SENTINEL = None\nPTX_OBS = re.compile\"")"
# U1 IS THE ROW THAT MATTERS: the state this increment starts from.  The file
# had no __main__ guard, so `import unroll` ran a whole-corpus census -- which
# is WHY the frontier never crossed this layer.
# U1 AS FIRST WRITTEN WAS MIS-AIMED and is kept, labelled, because the reason
# is the point: removing the guard alone leaves `factor`/`census` in place, so
# `frontier` still imports and calls them and the module merely prints a table
# on import.  The ORIGINAL file had NO importable API -- two helpers and
# top-level code -- which is what made the layer uncrossable, so U1b takes the
# API away and that is the row that reproduces the starting state.
row "U1 the main guard alone (MIS-AIMED, see U1b)" unroll.py "$(sub "\"if __name__ == '__main__':\"" "\"if True:\"")"
row "U1b unroll.py exposes no importable API (ORIGINAL)" unroll.py "$(sub "\"def factor(kernel, d='corpus'):\"" "\"def _no_such_api(kernel, d='corpus'):\"")"
row "U2 loop 0 paired with loop 0 (the x55 bug)" unroll.py "$(sub "\"    if ps != ss:\"" "\"    if False:\"")"
row "U3 a level that SHRANK read as a ratio" unroll.py "$(sub "\"    if any(r < 1 - 1e-9 for r in seen):\"" "\"    if False:\"")"
row "U4 a vacuous level read as undecidable" unroll.py "$(sub "\"        if a == 0 and b == 0:\"" "\"        if False:\"")"
row "U5 the async copy family is not observable" unroll.py "$(sub "\"SASS_OBS = re.compile(r'^(?:@!?P\\\\d+\\\\s+)?(?:LDG|STG|LDGSTS)\\\\b')\"" "\"SASS_OBS = re.compile(r'^(?:@!?P\\\\d+\\\\s+)?(?:LDG|STG)\\\\b')\"")"
row "U6 the frontier stops crossing the layer" frontier.py "$(sub "\"        if v == 'UNROLLED':\"" "\"        if False:\"")"
row "U7 the clear-set consistency control removed" frontier.py "$(sub "\"    bad = {k: _UNROLL[k] for k in clear\"" "\"    bad = {} or {k: _UNROLL[k] for k in []\"")"
row "U8 unroll_unknown re-derives instead of reading" frontier.py "$(sub "\"    missing = [k for k in ks if k not in _UNROLL]\"" "\"    return {k: unroll.factor(k)[1] for k in ks if unroll.factor(k)[0] in ('REFUSED','UNDECIDED')}\n    missing = [k for k in ks if k not in _UNROLL]\"")"
row "U9 the ground-truth check is dropped" docgate.py "$(sub "\"    for k, d, want in unroll.GROUND_TRUTH:\"" "\"    for k, d, want in []:\"")"
row "U10 the doc's unroll tally goes stale" ../../docs/ptxas_translation_validation.md "$(sub "\"| **24** |\"" "\"| **23** |\"")"
row "U7b clear-set control removed AND a clear kernel unrolled" frontier.py "$(printf "%s" "a=\"    bad = {k: _UNROLL[k] for k in clear\"
b=\"    bad = {} or {k: _UNROLL[k] for k in []\"
assert s.count(a)==1, 'anchor'
s=s.replace(a,b)
u=open('unroll.py').read()
ua=\"def factor(kernel, d='corpus'):\"
ub=\"def factor(kernel, d='corpus'):\\n    if kernel == 'bn254_fr_mul_fast': return 'UNROLLED', 'probe'\"
assert u.count(ua)==1, 'anchor2'
open('unroll.py','w').write(u.replace(ua,ub))")"
row "U9b ground truth dropped from docgate AND unroll" docgate.py "$(printf "%s" "a=\"    for k, d, want in unroll.GROUND_TRUTH:\"
b=\"    for k, d, want in []:\"
assert s.count(a)==1, 'anchor'
s=s.replace(a,b)
u=open('unroll.py').read()
ua=\"    for k, d, want in GROUND_TRUTH:\"
ub=\"    for k, d, want in []:\"
assert u.count(ua)==1, 'anchor2'
open('unroll.py','w').write(u.replace(ua,ub))")"
# U9 and U9b are both GREEN and both are confirmations rather than holes:
# removing one checker leaves the other enforcing the property, and removing
# BOTH is still green because with correct code there is nothing to catch --
# a guard removal is invisible unless the hazard exists.  U9c creates the
# hazard: it makes the EXPECTATION disagree with the measurement, which is the
# only way to see whether the assertion is live at all.
row "U9c the ground-truth expectation itself is wrong" unroll.py "$(sub "\"('exact_pv', 'corpus', 'UNROLLED')\"" "\"('exact_pv', 'corpus', 'MATCHED')\"")"
row "U12 the README's copy of the figures goes stale" ../../README.md "$(sub "\"103 distinct blockers and 6 clear\"" "\"104 distinct blockers and 6 clear\"")"
row "U11 OVER-REFUSAL: every kernel is UNROLLED" unroll.py "$(sub "\"    ratios, vacuous = [], 0\"" "\"    return 'UNROLLED', 'probe: everything'\n    ratios, vacuous = [], 0\"")"
row "BASE again" '' ''
