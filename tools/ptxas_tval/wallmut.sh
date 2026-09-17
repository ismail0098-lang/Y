#!/bin/bash
# Mutation table for the FIFTH LAYER -- the solver-wall proxy (`wall.py`) and its
# crossing into the frontier.
#
# One row per mutation.  Each row restores the baseline, applies ONE patch
# (asserted to apply exactly once, or the row says so rather than running an
# unmutated tree), and prints one letter per check: . pass, F fail.
#
#   WALLMUT_ONLY=W3   run only the rows whose label contains that text
#
# READ THE CONTROL ROW FIRST and the closing BASE row second: a table whose
# control is red reports the state of the tree, not the mutation.
#
# THE `docgate` COLUMN IGNORES THE STAMP-TREE FAILURE, for the reason `sumut.sh`
# and `unrmut.sh` record: `wall.py`, `barregion.py` and `frontier.py` are all in
# `frontier.MODELS`, so any edit to one invalidates the stamp by construction.
# That is the stamp working, and it would make the column red on the control row.
#
# Run it from a FILE, never concurrently with `frontier.py`.
cd "$(dirname "$0")"
rm -rf __pycache__
BASE=$(mktemp -d)
FILES="wall.py barregion.py frontier.py docgate.py"
for f in $FILES; do cp "$f" "$BASE/"; done
cp ../../docs/ptxas_translation_validation.md "$BASE/tvaldoc.md"
restore() {
  for f in $FILES; do cp "$BASE/$f" .; touch "$f"; done
  cp "$BASE/tvaldoc.md" ../../docs/ptxas_translation_validation.md
  rm -rf __pycache__
}
trap 'restore; rm -rf "$BASE"' EXIT

checks() {
  local s=""
  timeout 300 python3 wall.py     --selftest >/dev/null 2>&1 && s="$s." || s="${s}F"
  timeout 900 python3 frontier.py --selftest >/dev/null 2>&1 && s="$s." || s="${s}F"
  local out
  out=$(timeout 1800 python3 docgate.py 2>&1)
  if printf '%s\n' "$out" | sed '/--- positive controls/,$d' \
       | grep '^FAIL:' | grep -qv 'stamped on a different tree'
  then s="${s}F"; else s="$s."; fi
  echo "$s"
}

row() {
  local label="$1" file="$2" py="$3"
  if [ -n "$WALLMUT_ONLY" ] && [[ "$label" != *"$WALLMUT_ONLY"* ]]; then return; fi
  restore
  if [ -n "$py" ]; then
    if ! python3 - "$file" <<PY
import sys
f = sys.argv[1]; s = open(f).read()
$py
open(f, 'w').write(s)
PY
    then printf '%-54s MUTATION DID NOT APPLY\n' "$label"; return; fi
  fi
  printf '%-54s %s\n' "$label" "$(checks)"
}

sub() { printf 'a=%s\nb=%s\nassert s.count(a)==1, "anchor"\ns=s.replace(a,b)\n' "$1" "$2"; }

printf '%-54s %s\n' '' 'wall / frontier / docgate'
row "BASE" '' ''
row "W0 CONTROL two independent assignments reordered" wall.py "$(sub "\"_here = os.path.dirname(os.path.abspath(__file__))\n_cwd = os.getcwd()\"" "\"_cwd = os.getcwd()\n_here = os.path.dirname(os.path.abspath(__file__))\"")"
# W1 IS THE ROW THAT MATTERS: the state this increment starts from.  The layer
# is measured and never crossed, so every clear-but-unvalidated kernel reads
# clear again.
row "W1 the frontier stops crossing the wall (ORIGINAL)" frontier.py "$(sub "\"        if v == 'PAST':\"" "\"        if False:\"")"
row "W2 thresholds transcribed as the old 29 / 65" wall.py "$(sub "\"    UNDER_AT, PAST_AT = _thresholds()\"" "\"    UNDER_AT, PAST_AT = 29, 65\"")"
row "W3 every integer multiply counts, constants too" wall.py "$(sub "\"    return len(ops) >= 3 and all(o.startswith('%') for o in ops[1:3])\"" "\"    return len(ops) >= 3\"")"
row "W4 fma is not a multiply (barregion reverted)" barregion.py "$(sub "\"o.startswith(('mul.', 'mad.', 'fma.'))\"" "\"o.startswith(('mul.', 'mad.'))\"")"
row "W5 the entry pattern reverted (split kernels PAST)" wall.py "$(sub "\"r'^\\\\s*(?:\\\\.visible\\\\s+)?\\\\.entry\\\\b'\"" "\"r'^\\\\s*\\\\.(?:visible\\\\s+)?entry\\\\b'\"")"
row "W6 the region-level UNKNOWN point dropped" wall.py "$(sub "\"    ('bn254_ntt4_fused',  1, 'UNKNOWN',\"" "\"    ('bn254_ntt4_fused',  0, 'PROVED',\"")"
row "W7 OVER-REFUSAL: every kernel is PAST" wall.py "$(sub "\"    w = max((t for t, _i in m), default=0)\"" "\"    return 'PAST', 'probe: everything'\n    w = max((t for t, _i in m), default=0)\"")"
row "W8 regress_validated reads nothing (its floor)" wall.py "$(sub "\"re.findall(r'\\\\b(corpus|smut|o1)/(\\\\w+)', s)\"" "\"re.findall(r'\\\\b(nowhere)/(\\\\w+)', s)\"")"
row "W9 barregion's CLI guard removed" barregion.py "$(sub "\"if __name__ == '__main__':\"" "\"if True:\"")"
row "W10 float-driven region called PAST" wall.py "$(sub "\"    if w >= PAST_AT:\n        return 'UNDECIDED'\"" "\"    if w >= PAST_AT:\n        return 'PAST'\"")"
row "W11 the frontier's wall biconditional removed" frontier.py "$(sub "\"        if has != (wsnap[k][0] == 'PAST'):\"" "\"        if False:\"")"
row "W11b biconditional removed AND the crossing removed" frontier.py "$(printf "%s" "a=\"        if has != (wsnap[k][0] == 'PAST'):\"
b=\"        if False:\"
assert s.count(a)==1, 'anchor'
s=s.replace(a,b)
a=\"        if v == 'PAST':\"
b=\"        if False:\"
assert s.count(a)==1, 'anchor2'
s=s.replace(a,b)")"
row "W12 the doc's wall bracket goes stale" ../../docs/ptxas_translation_validation.md "$(sub "\"solver wall between **33** and **49**\"" "\"solver wall between **29** and **49**\"")"
row "W13 the doc's -O3 one-away set drops a kernel" ../../docs/ptxas_translation_validation.md "$(sub "\"\`bn254_g1_dbl\` and\n\`bn254_ntt4_fused\` — each has\"" "\"\`bn254_g1_dbl\` — each has\"")"
row "BASE again" '' ''
