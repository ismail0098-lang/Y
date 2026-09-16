#!/bin/bash
# Mutation table for the SUITE DISPATCH and the decorative-row gate.
#
# One row per mutation.  Each row restores the baseline, applies ONE patch
# (asserted to apply exactly once, or the row says so rather than running an
# unmutated tree -- which is the defect `mutgate.py` exists for, found in
# `rmut.sh`), and prints one letter per check: . pass, F fail.
#
#   SUMUT_ONLY=S3   run only the rows whose label contains that text
#
# READ THE CONTROL ROW FIRST and the closing BASE row second: a table whose
# control is red reports the state of the tree, not the mutation.
#
# THE `docgate` COLUMN IGNORES THE STAMP-TREE FAILURE, and that is not a
# weakening.  `loopgap.py` and `frontier.py` are in `frontier.MODELS`, so ANY
# edit to either invalidates the stamp by construction -- which is the stamp
# working, and it made the column red on the CONTROL row, where it can
# distinguish nothing.  A column that fails on a no-op is not a column.  Every
# other `docgate` failure, including the census's own figures, is graded.
#
# Run it from a FILE, never concurrently with `frontier.py` -- its structural
# child imports these modules fresh.
cd "$(dirname "$0")"
rm -rf __pycache__
BASE=$(mktemp -d)
FILES="loopgap.py frontier.py docgate.py fpgate.py mutgate.py rmut.sh"
DOCS="../../docs/ptxas_translation_validation.md ../../README.md"
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
  timeout 600  python3 loopgap.py --selftest >/dev/null 2>&1 && s="$s." || s="${s}F"
  timeout 600  python3 mutgate.py           >/dev/null 2>&1 && s="$s." || s="${s}F"
  timeout 1800 python3 fpgate.py            >/dev/null 2>&1 && s="$s." || s="${s}F"
  local out
  out=$(timeout 1800 python3 docgate.py 2>&1)
  if printf '%s\n' "$out" | sed '/--- positive controls/,$d' \
       | grep '^FAIL:' | grep -qv 'stamped on a different tree'
  then s="${s}F"; else s="$s."; fi
  echo "$s"
}

row() {
  local label="$1" file="$2" py="$3"
  if [ -n "$SUMUT_ONLY" ] && [[ "$label" != *"$SUMUT_ONLY"* ]]; then return; fi
  restore
  if [ -n "$py" ]; then
    if ! python3 - "$file" <<PY
import sys
f = sys.argv[1]; s = open(f).read()
$py
open(f, 'w').write(s)
PY
    then printf '%-46s MUTATION DID NOT APPLY\n' "$label"; return; fi
  fi
  printf '%-46s %s\n' "$label" "$(checks)"
}

sub() { printf 'a=%s\nb=%s\nassert s.count(a)==1, "anchor"\ns=s.replace(a,b)\n' "$1" "$2"; }

printf '%-46s %s\n' '' 'selftest / mutgate / fpgate / docgate'
row "BASE" '' ''
row "S0 CONTROL reorder two independent lines" loopgap.py "$(sub "\"    p, s = f'{d}/{k}.ptx', f'{d}/{k}.sass'\n    try:\"" "\"    p = f'{d}/{k}.ptx'\n    s = f'{d}/{k}.sass'\n    try:\"")"
row "S1 the census asks loopval alone (ORIGINAL)" loopgap.py "$(sub "\"    if not BACKEDGE.search(lmsg.split('\\\\n')[0]):\"" "\"    if True:\"")"
row "S2 nestval asked for EVERY loopval refusal" loopgap.py "$(sub "\"    if not BACKEDGE.search(lmsg.split('\\\\n')[0]):\"" "\"    if False:\"")"
row "S3 counts no longer folded" loopgap.py "$(sub "\"    m = TOPLEVEL.search(msg) or BACKEDGE.search(msg)\"" "\"    m = None\n    if TOPLEVEL.search(msg): return re.sub(r'\\\\s+', ' ', msg)\n    m = BACKEDGE.search(msg)\"")"
row "S4 'no loop' folded into more-than-one" loopgap.py "$(sub "\"        return f'{m.group(1)}: loop finder found NO back edge'\"" "\"        return f'{m.group(1)}: more than one loop at one level, shape unknown'\"")"
# S5 as first written mislabelled only the SUCCESS return, and at -O3 every
# nestval answer comes back through the `except` -- so it changed nothing and
# read as a survivor.  Both returns now.
row "S5 who mislabelled as loopval throughout" loopgap.py "$(printf "%s" "for a,b in ((\"return 'nestval', v, msg, n\", \"return 'loopval', v, msg, n\"), (\"return 'nestval', 'REFUSED', str(e), 0\", \"return 'loopval', 'REFUSED', str(e), 0\")):
    assert s.count(a)==1, 'anchor'
    s=s.replace(a,b)")"
row "S6 mutgate applies no step (floor)" mutgate.py "$(sub "\"    for m in HEREDOC.finditer(harness_src):\"" "\"    for m in []:\"")"
# S7 as first written removed R2's assertion and left its substitutions, so the
# file still changed and mutgate was right to call the step live.  Breaking an
# ANCHOR is the probe: the assert fires, nothing is written, the row runs an
# unmutated tree -- which is the defect this gate exists for.
row "S7 rmut.sh R2's anchor broken again" rmut.sh "$(sub "\"for a, b in ((\\\"        return f'{m.group(1)}: loop finder found NO back edge'\\\",\"" "\"for a, b in ((\\\"        return f'{m.group(1)}: a string that is not in the file'\\\",\"")"
row "S8 mutgate reports no no-ops" mutgate.py "$(sub "\"            bad.append((h, kind, target, 'the patch changes nothing'))\"" "\"            pass\"")"
row "S9 fpgate back to its own dispatch" fpgate.py "$(sub "\"            _who, v, msg, _n = loopgap.suite_validate(k, 20, 'wide', d)\"" "\"            v, msg, _ = loopval.validate(p, s, 20, 'wide')\"")"
row "S10 OVER-REFUSAL: the suite refuses always" loopgap.py "$(sub "\"    p, s = f'{d}/{k}.ptx', f'{d}/{k}.sass'\n    try:\"" "\"    p, s = f'{d}/{k}.ptx', f'{d}/{k}.sass'\n    return 'loopval', 'REFUSED', 'probe: refuse everything', 0\n    try:\"")"
# S11's first form carried backticks through two layers of shell quoting and
# did not apply -- the harness said so instead of running an unmutated tree.
row "S11 the doc drops the -O1 sole-blocker claim" ../../docs/ptxas_translation_validation.md "$(sub "\"**no kernel is one\nblocker away**\"" "\"the sole-blocker set is what it is\"")"
row "S12 the doc's -O3 distinct figure goes stale" ../../docs/ptxas_translation_validation.md "$(sub "\"**104** distinct blockers, and not one of them\"" "\"**103** distinct blockers, and not one of them\"")"
row "S13 the README's copy of the census goes stale" ../../README.md "$(sub "\"**25 of the 48\nrefuse for one reason: more than one loop at one level** (25 on the PTX side, 0\"" "\"**38 of the 48\nrefuse for one reason: more than one loop at one level** (38 on the PTX side, 0\"")"
row "BASE again" '' ''
