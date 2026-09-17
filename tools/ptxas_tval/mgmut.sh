#!/bin/bash
# Mutation table for the DECORATIVE-ROW GATE itself, after it grew from one
# harness style to four.
#
# One row per mutation.  Each row restores the baseline, applies ONE patch
# (asserted to apply exactly once, or the row says so rather than running an
# unmutated tree -- which is the defect this gate exists for), and prints one
# letter per check: . pass, F fail.
#
#   MGMUT_ONLY=M3   run only the rows whose label contains that text
#
# READ THE CONTROL ROW FIRST and the closing BASE row second: a table whose
# control is red reports the state of the tree, not the mutation.
#
# THERE IS NO VALIDATOR COLUMN, and that is a structural fact rather than an
# omission: `mutgate.py` is imported by `docgate.py` and by nothing else, and it
# is not in `frontier.MODELS`, so no probe here can move a verdict, an
# obligation count or the census stamp.  `liftgap --selftest` is carried as the
# independent column that says so.
#
# THIS HARNESS IS ITSELF STYLE 4, so the gate under test dry-runs it, which is
# the cheapest available check that the dry run works on a harness nobody had
# written when it was designed -- and it is also why the `mutgate` column has to
# ignore this table's own rows; see `checks` below.
cd "$(dirname "$0")"
rm -rf __pycache__
BASE=$(mktemp -d)
FILES="mutgate.py sumut.sh"
for f in $FILES; do cp "$f" "$BASE/"; done
restore() {
  for f in $FILES; do cp "$BASE/$f" .; touch "$f"; done
  rm -rf __pycache__
}
trap 'restore; rm -rf "$BASE"' EXIT

checks() {
  local s=""
  # THE `mutgate` COLUMN IGNORES THIS TABLE'S OWN ROWS, and that is forced
  # rather than chosen.  A mutation harness is an INPUT to this gate: every row
  # here patches `mutgate.py` or `sumut.sh`, the gate then re-applies that same
  # row to the already-patched tree, its anchor is gone, and the gate reports it
  # as decorative -- correctly, and on EVERY row including the control, which
  # made the whole column red and carried no information.  Every FAIL naming any
  # OTHER harness, the four controls, the floor and the count agreement are all
  # still graded.  Same shape as `sumut.sh`'s stamp-tree exclusion.
  local mg
  mg=$(timeout 600 python3 mutgate.py 2>&1)
  if printf '%s\n' "$mg" | grep '^FAIL:' | grep -qv '^FAIL: mgmut.sh:'
  then s="${s}F"; else s="$s."; fi
  # THE `docgate` COLUMN GRADES THE DOC FIGURES, NOT mutgate's VERDICT.
  # `docgate` does enforce it -- `bad += mutgate.main()` then `sys.exit(1)`,
  # verified by breaking `differs` and reading rc=1 -- but mutgate runs after the
  # positive-controls separator that this grading cuts, so what stays in the
  # column is the independent claim that none of these probes moves a published
  # figure.  A column that merely mirrored the first one would say nothing.
  local out
  out=$(timeout 1800 python3 docgate.py 2>&1)
  if printf '%s\n' "$out" | sed '/--- positive controls/,$d' \
       | grep '^FAIL:' | grep -qv 'stamped on a different tree'
  then s="${s}F"; else s="$s."; fi
  timeout 600  python3 liftgap.py --selftest >/dev/null 2>&1 && s="$s." || s="${s}F"
  echo "$s"
}

row() {
  local label="$1" file="$2" py="$3"
  if [ -n "$MGMUT_ONLY" ] && [[ "$label" != *"$MGMUT_ONLY"* ]]; then return; fi
  restore
  if [ -n "$py" ]; then
    if ! python3 - "$file" <<PY
import sys
f = sys.argv[1]; s = open(f).read()
$py
open(f, 'w').write(s)
PY
    then printf '%-52s MUTATION DID NOT APPLY\n' "$label"; return; fi
  fi
  printf '%-52s %s\n' "$label" "$(checks)"
}

sub() { printf 'a=%s\nb=%s\nassert s.count(a)==1, "anchor"\ns=s.replace(a,b)\n' "$1" "$2"; }

printf '%-52s %s\n' '' 'mutgate / docgate / liftgap'
row "BASE" '' ''
row "M0 CONTROL two independent copies reordered" mutgate.py "$(sub "\"    shutil.copytree(os.path.join(ROOT, 'docs'), os.path.join(dst, 'docs'))\n    shutil.copy(os.path.join(ROOT, 'README.md'), os.path.join(dst, 'README.md'))\"" "\"    shutil.copy(os.path.join(ROOT, 'README.md'), os.path.join(dst, 'README.md'))\n    shutil.copytree(os.path.join(ROOT, 'docs'), os.path.join(dst, 'docs'))\"")"
row "M1 a heredoc must follow python3 - (ORIGINAL)" mutgate.py "$(sub "\"    for m in HEREDOC.finditer(harness_src):\n        if WRITES.search(m.group(2)):\"" "\"    for m in HEREDOC.finditer(harness_src):\n        if WRITES.search(m.group(2)) and harness_src[max(0, m.start() - 12):m.start()].rstrip().endswith('python3 -'):\"")"
row "M2 the per-row script style is not recovered" mutgate.py "$(sub "\"    d = SCRIPTDIR.search(harness_src)\"" "\"    d = None\"")"
row "M3 style-4 rows are never dry-run" mutgate.py "$(sub "\"    return names[0] if len(names) == 1 else None\"" "\"    return None\"")"
row "M4 the stub emits no token (uptake control)" mutgate.py "$(sub '"  else echo \"MUTGATE-LIVE\"; fi"' '"  else echo \"\"; fi"')"
row "M5 a baseline row counted as a patch" mutgate.py "$(sub '"  if [ -z \"$file\" ] || [ -z \"$py\" ]; then echo \"MUTGATE-NOPATCH\"; return; fi"' '"  if [ -z \"$file\" ]; then echo \"MUTGATE-NOPATCH\"; return; fi"')"
row "M6 an uninvoked patching function is ignored" mutgate.py "$(sub "\"    out = []\n    for name in FUNCDEF.findall(harness_src):\"" "\"    return []\n    for name in FUNCDEF.findall(harness_src):\"")"
row "M7 the one-line brace matcher reverted" mutgate.py "$(sub "\"        body = harness_src[i:i + _extent(harness_src, i)]\"" "\"        body = harness_src[i:harness_src.find(chr(10) + '}' + chr(10), i) + 3] or harness_src[i:]\"")"
row "M8 nothing is ever seen to change" mutgate.py "$(sub "\"    return subprocess.run(['diff', '-rq', a, b], capture_output=True).returncode != 0\"" "\"    return True\"")"
row "M9 the sandbox omits the docs a row patches" mutgate.py "$(sub "\"    shutil.copytree(os.path.join(ROOT, 'docs'), os.path.join(dst, 'docs'))\"" "\"    os.makedirs(os.path.join(dst, 'docs'))\"")"
row "M10 OVER-REFUSAL every step is decorative" mutgate.py "$(sub "\"                if differs(work, base):\"" "\"                if False:\"")"
row "M11 sumut S8's anchor broken again" sumut.sh "$(sub "\"h, kind, what,\"" "\"h, kind, target,\"")"
row "BASE again" '' ''
