"""Every mutation row's patch must CHANGE the file it targets, in EVERY harness.

A mutation table is the only evidence in this directory that a check is
load-bearing, and a row whose patch cannot apply runs an UNMUTATED tree and
reports a perfectly plausible line.  That is not hypothetical: `memmut.sh` had
seven such rows when the effect model landed, and this file was written after
`rmut.sh`'s R2 -- "collapse ZERO back edges into the more-than-one bucket" --
turned out to have been decorative since the increment that rewrote
`loopgap.reason_key` underneath it.  Nobody re-runs an old table, so the decay
is silent by construction.

IT USED TO COVER 3 OF 14 HARNESSES, which is the null-metric shape in the gate
written to prevent it: "22 patch steps, every one of them changes the file it
targets" reads as a statement about the harnesses and was a statement about the
ones whose steps it could recover.  It recognised ONE spelling -- a
`python3 - <<'EOF'` heredoc naming a literal target -- and there are four:

  1. `python3 - <<'EOF' ... EOF` / `sed -i ... FILE.py`   gmut, lmut, rmut
  2. a heredoc consumed by ANY command, the target named inside the body
     (`M <<'P'`, where `M` pipes stdin to `python3 -`)     memmut, nmut, rtmut, xmut
  3. a standalone script per row (`python3 muts/$m.py`)    smut
  4. `row "label" TARGET "$(sub ...)"`, the payload built by SHELL expansion
                                                 nestmut, sumut, unrmut, mgmut

1-3 are recovered statically because their payload is verbatim.  4 cannot be:
the anchors are shell-quoted regexes inside shell-quoted strings, and a scanner
that re-derives what the shell will see gets it wrong in both directions -- the
first version of this measurement reported four false stale anchors in
`lmut.sh` for exactly that reason, and running them found all four live.  So
style 4 is EXECUTED: the harness is sourced in a sandbox with its own row
runner stubbed out, which makes the shell do every expansion and the harness's
own `assert s.count(a)==1` fire, in about a quarter of a second instead of the
hours the real table costs.

WHAT IT CHECKS, and it is the PROPERTY rather than the plumbing.  The obvious
rule is "every patch must assert that it applied", which is a rule about
whether somebody wrote an assertion -- measured when this was written, nine of
the nine file-writing heredocs in the directory had none.  This applies each
step in a SANDBOX COPY of the directory and requires the tree to differ.  A row
that asserts nothing is fine as long as its patch bites; a row that asserts and
whose anchor has rotted is caught anyway.  One property, one rule, all four
styles -- a gate that checked "the assert exists" for some and "the file moved"
for others would be two rules wearing one name.

THE SANDBOX IS THE WHOLE DIRECTORY, not the one file a payload names, because a
step may write a FIXTURE (`smut/smem_roundtrip.sass`), may write several files,
and may compose two sibling scripts it reads at run time (`muts/S1b.py` is
`exec(open('muts/S1c.py'))` + `exec(open('muts/S1.py'))`, and naming its target
statically is impossible).  `corpus/` and `o1/` are left out -- 14 MB, and
measured: no step in any harness writes into either.

COMPOUND ROWS.  Some rows apply a second patch on top of a first, so a step can
legitimately be a no-op against a pristine tree and bite against the cumulative
one.  A step is live if it changes EITHER, which errs towards silence; the
alternative is an allowlist, which is the defect this directory keeps finding.

A HARNESS WITH NO ROWS IS THE WORSE DEFECT, and closing the coverage gap is
what surfaced one: `fpmut.sh` defined a `probe()` that applies a patch, called
it ZERO times, and had done so since the commit that moved this directory into
the repository.  It looked like a table -- a runner, a restore, the
"a surviving archive means this run did not finish" protocol -- and running it
printed nothing and exited 0, which is indistinguishable from a clean table.
So every harness must be classified, and `no rows` is a FAILURE unless the
harness is a RUNNER that delegates to another one or takes its patch from its
own arguments.  Those two are properties of the text, not an allowlist.

CONTROLS.  A floor (a scan that applied no step reports no no-ops perfectly);
two positive ones through the same `apply` the real scan uses, a dead heredoc
anchor and a dead sed pattern; one that a live step is NOT reported; and, for
style 4, the stub's own uptake -- every row a harness prints must carry a token
only the stub can emit, so a runner this file failed to find is a loud failure
rather than a silent full-speed run of the real table.
"""
import glob, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))

# A heredoc consumed by ANY command -- `python3 - <<'X'`, or a shell function
# that pipes stdin into one.  What makes it a PATCH is that its body writes a
# named file, which is what `WRITES` decides.
HEREDOC = re.compile(r"<<\s*'(\w+)'\n(.*?)\n\1\n", re.S)
WRITES = re.compile(r"open\((['\"])([\w./]+)\1\s*,\s*['\"]w")
SEDLINE = re.compile(r"(?m)^.*\bsed\s+-i\b.*$")
SEDFILE = re.compile(r"([A-Za-z0-9_]+\.py)\s*$")
# `python3 "muts/$m.py"` -- one script per row, in a directory of its own.
SCRIPTDIR = re.compile(r'python3\s+"?([a-z_]+)/\$\w+\.py')
FUNCDEF = re.compile(r"(?m)^([a-z_][a-z_0-9]*)\s*\(\)\s*\{")
# `$(checks)` / `$(verdicts)`: a command substitution calling a function with no
# arguments, which is how all three style-4 harnesses invoke their row runner.
NOARGSUB = re.compile(r'\$\(([a-z_][a-z_0-9]*)\)')
DELEGATES = re.compile(r'\./\w*mut\w*\.sh')
ROWLINE = re.compile(r'(?m)^row\s')
APPLIES = re.compile(r"<<\s*'\w+'|sed\s+-i|python3\s+-c")


def steps(harness_src, read=None):
    """(kind, payload) for every statically recoverable patch step, in order."""
    out = []
    for m in HEREDOC.finditer(harness_src):
        if WRITES.search(m.group(2)):
            out.append(('heredoc', m.group(2), m.start()))
    for m in SEDLINE.finditer(harness_src):
        line = m.group(0)
        if SEDFILE.search(line):
            out.append(('sed', line, m.start()))
    d = SCRIPTDIR.search(harness_src)
    if d:
        lister = read or (lambda p: sorted(
            os.path.relpath(x, HERE) for x in glob.glob(os.path.join(HERE, p, '*.py'))))
        for rel in lister(d.group(1)):
            out.append(('script', rel, len(harness_src) + len(out)))
    out.sort(key=lambda s: s[2])
    return [(k, p) for k, p, _ in out]


def patch_rows(harness_src):
    """The `row` invocations that carry a patch.

    A BASELINE row passes an empty payload -- `row "BASE" '' ''`, or
    `row "BASE" nestval.py \"\"` -- and applies nothing, so counting it as a
    patch step inflates the headline, which is the null-metric shape this
    file exists to remove.
    """
    out = []
    for line in harness_src.splitlines():
        if not line.startswith('row '):
            continue
        last = line.split()[-1]
        if last not in ("''", '""'):
            out.append(line)
    return out


def row_runner(harness_src):
    """The function a style-4 harness calls to fill its verdict column."""
    defined = set(FUNCDEF.findall(harness_src))
    names = sorted(set(NOARGSUB.findall(harness_src)) & defined)
    return names[0] if len(names) == 1 else None


def _extent(src, i):
    """How far a shell function definition at `i` runs, by matching its braces.

    A one-line `f(){ cmd; }` has no closing brace on its own line, so a search
    for "\\n}\\n" walks off the end of the file and reports every function as a
    patcher -- which is what `fpmut.sh`'s three one-liners did on the first run.
    """
    depth, j = 0, src.index('{', i)
    start = j
    while j < len(src):
        if src[j] == '{':
            depth += 1
        elif src[j] == '}':
            depth -= 1
            if depth == 0:
                return j - i + 1
        j += 1
    return len(src) - i


def uninvoked_patchers(harness_src):
    """Functions that apply a patch and are never called -- a table's rows gone."""
    out = []
    for name in FUNCDEF.findall(harness_src):
        i = harness_src.index(f'{name}()')
        body = harness_src[i:i + _extent(harness_src, i)]
        if not APPLIES.search(body):
            continue
        rest = harness_src[:i] + harness_src[i + len(body):]
        if not re.search(rf'(?m)^\s*{re.escape(name)}\b', rest):
            out.append(name)
    return out


def _mkroot(seed=None):
    """A sandbox: the tval directory, the docs and the README a row may patch."""
    d = tempfile.mkdtemp(prefix='mutgate_root_')
    dst = os.path.join(d, 'root')
    tval = os.path.join(dst, 'tools', 'ptxas_tval')
    os.makedirs(os.path.dirname(tval))
    shutil.copytree(HERE, tval, symlinks=True,
                    ignore=shutil.ignore_patterns('corpus', 'o1', '__pycache__'))
    shutil.copytree(os.path.join(ROOT, 'docs'), os.path.join(dst, 'docs'))
    shutil.copy(os.path.join(ROOT, 'README.md'), os.path.join(dst, 'README.md'))
    for name, text in (seed or {}).items():
        with open(os.path.join(tval, name), 'w') as fh:
            fh.write(text)
    return d, dst


def apply(kind, payload, base_root):
    """Run one step over a COPY of `base_root`; return the copy."""
    d = tempfile.mkdtemp(prefix='mutgate_')
    work = os.path.join(d, 'root')
    shutil.copytree(base_root, work, symlinks=True)
    cwd = os.path.join(work, 'tools', 'ptxas_tval')
    cmd = {'heredoc': [sys.executable, '-c', payload],
           'script': [sys.executable, payload],
           'sed': ['bash', '-c', payload]}[kind]
    try:
        subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        pass
    return work


def differs(a, b):
    return subprocess.run(['diff', '-rq', a, b], capture_output=True).returncode != 0


STUB = r'''
__mutgate_stub(){
  if [ -z "$file" ] || [ -z "$py" ]; then echo "MUTGATE-NOPATCH"; return; fi
  local rel; rel=$(realpath -m --relative-to="$MUTGATE_ROOT" "$file")
  if cmp -s "$file" "$MUTGATE_PRISTINE/$rel"; then echo "MUTGATE-NOOP"
  else echo "MUTGATE-LIVE"; fi
}
'''


def dry_run(harness, src, runner, base_root, keep):
    """Source a style-4 harness with its row runner stubbed; return its rows.

    `readonly -f` is what makes this need no edit to any harness: bash REFUSES
    the harness's own definition of the runner and carries on, so the stub
    survives and nothing the real runner would have done is run.  The stub sees
    `$file` because a row calls it through `$(...)`, whose subshell inherits the
    row's locals.
    """
    d = tempfile.mkdtemp(prefix='mutgate_dry_')
    keep.append(d)
    work = os.path.join(d, 'root')
    shutil.copytree(base_root, work, symlinks=True)
    stub = os.path.join(d, 'stub.sh')
    with open(stub, 'w') as fh:
        fh.write(STUB + f'{runner}(){{ __mutgate_stub; }}\nreadonly -f {runner}\n')
    env = dict(os.environ, MUTGATE_ROOT=work, MUTGATE_PRISTINE=base_root)
    for v in list(env):
        if v.endswith('_ONLY'):
            del env[v]
    r = subprocess.run(
        ['bash', '-c', 'source "$2"; source "$1"', 'bash', f'./{harness}', stub],
        cwd=os.path.join(work, 'tools', 'ptxas_tval'), env=env,
        capture_output=True, text=True, timeout=600)
    rows = [l for l in r.stdout.splitlines()
            if 'MUTGATE-' in l or 'MUTATION DID NOT APPLY' in l]
    # UPTAKE CONTROL.  A runner this file failed to stub would have run the real
    # table and printed a plausible verdict column, so the row count is what says
    # the stub took.
    want = len(ROWLINE.findall(src))
    if len(rows) != want:
        return None, (f'the stub for `{runner}` printed {len(rows)} rows where the '
                      f'harness invokes `row` {want} times, so it did not take')
    return rows, None


def scan(harnesses=None, read=None, lister=None, seed=None):
    """(applied, [(harness, kind, what, why)]) for every decorative step."""
    read = read or (lambda p: open(os.path.join(HERE, p)).read())
    keep = []
    rootdir, base = _mkroot(seed)
    keep.append(rootdir)
    bad, applied = [], 0
    try:
        for h in harnesses if harnesses is not None else all_harnesses():
            try:
                src = read(h)
            except OSError:
                continue
            runner = row_runner(src)
            if runner is not None:
                # A harness that mixed the two would have its static steps
                # silently dropped.  None does today, so the case is REPORTED
                # rather than handled -- machinery for a case that cannot arise
                # is machinery nothing tests.
                extra = steps(src, lister)
                if extra:
                    bad.append((h, 'mixed', h, f'it has a row runner AND '
                                f'{len(extra)} statically recoverable step(s), '
                                f'which this scan would ignore'))
                rows, why = dry_run(h, src, runner, base, keep)
                if rows is None:
                    bad.append((h, 'dry-run', h, why))
                    continue
                for line in rows:
                    if 'MUTGATE-NOPATCH' in line:
                        continue  # a baseline row: it applies no patch
                    applied += 1
                    if 'MUTGATE-NOOP' in line:
                        bad.append((h, 'row', line.split('  ')[0].strip(),
                                    'the patch changes nothing'))
                    elif 'MUTATION DID NOT APPLY' in line:
                        bad.append((h, 'row', line.split('  ')[0].strip(),
                                    'the patch could not be applied'))
                continue
            cum = None
            for kind, payload in steps(src, lister):
                applied += 1
                work = apply(kind, payload, base)
                keep.append(os.path.dirname(work))
                what = payload if kind == 'script' else payload.splitlines()[0][:48]
                if differs(work, base):
                    cum = work
                    continue
                # a compound step may only bite on top of an earlier one
                if cum is not None:
                    w2 = apply(kind, payload, cum)
                    keep.append(os.path.dirname(w2))
                    if differs(w2, cum):
                        continue
                bad.append((h, kind, what, 'the patch changes nothing'))
    finally:
        for d in keep:
            shutil.rmtree(d, ignore_errors=True)
    return applied, bad


def all_harnesses():
    return sorted(os.path.basename(x)
                  for x in glob.glob(os.path.join(HERE, '*mut*.sh')))


def coverage(harnesses=None, read=None, lister=None):
    """Classify every harness: rows, or a NAMED reason it legitimately has none.

    `rows` is the count a static scan recovers, or the number of `row`
    invocations for a style-4 harness.  A harness with none must be a RUNNER --
    it invokes another harness, or it applies a patch that its caller supplied
    -- and anything else is a table that exercises nothing, which is the defect
    `fpmut.sh` had for its whole life.  Both exemptions are decided from the
    text; neither is a list of names.
    """
    read = read or (lambda p: open(os.path.join(HERE, p)).read())
    out = {}
    for h in harnesses if harnesses is not None else all_harnesses():
        try:
            src = read(h)
        except OSError:
            continue
        if row_runner(src) is not None:
            out[h] = ('rows', len(patch_rows(src)))
        elif steps(src, lister):
            out[h] = ('rows', len(steps(src, lister)))
        elif uninvoked_patchers(src):
            out[h] = ('EMPTY: defines %s, which applies a patch, and never '
                      'invokes it' % ', '.join(uninvoked_patchers(src)), 0)
        elif DELEGATES.search(src):
            out[h] = ('runner: delegates to another harness', 0)
        elif not FUNCDEF.findall(src) and APPLIES.search(src):
            out[h] = ('runner: applies the patch its caller supplies', 0)
        else:
            out[h] = ('EMPTY: no rows, and it is not a runner', 0)
    return out


def _controls():
    """Positive controls, through the same `apply`/`coverage` the real scan uses."""
    ok = True
    dead = ("s = open('victim.py').read()\n"
            "s = s.replace('a string that is not in the file', 'x')\n"
            "open('victim.py', 'w').write(s)\n")
    fake = ("python3 - <<'P'\n" + dead + "P\n"
            "sed -i 's/^a line that is not there$/x/' victim.py\n")
    files = {'fake_mut.sh': fake, 'victim.py': 'print(1)\n'}
    n, bad = scan(['fake_mut.sh'], read=lambda p: files[p],
                  seed={'victim.py': 'print(1)\n'})
    kinds = sorted(k for _h, k, _t, _w in bad)
    if n != 2 or kinds != ['heredoc', 'sed']:
        print(f'FAIL: the control harness has two dead steps and the scan '
              f'applied {n} and reported {kinds}')
        ok = False
    else:
        print('  control: a synthetic harness with a dead heredoc anchor and a '
              'dead sed pattern is reported, both kinds')
    live = ("python3 - <<'P'\n"
            "s = open('victim.py').read()\ns = s.replace('print(1)', 'print(2)')\n"
            "open('victim.py', 'w').write(s)\nP\n")
    n2, bad2 = scan(['live_mut.sh'], read=lambda p: {'live_mut.sh': live}[p],
                    seed={'victim.py': 'print(1)\n'})
    if n2 != 1 or bad2:
        print(f'FAIL: a step that DOES change its target was reported: {bad2}')
        ok = False
    else:
        print('  control: a step that changes its target is not reported')
    # A one-line `f(){ cmd; }` closes its brace on its own line, and an extent
    # search for "\n}\n" therefore runs to the end of the file -- which makes a
    # function that applies NO patch inherit the next one's `python3 -c` and be
    # reported as an uninvoked patcher.  `run` here is that function.
    oneline = ('run(){ echo hi; }\n'
               'probe(){ python3 -c "$1"; }\n'
               'probe "s=1"\n')
    if uninvoked_patchers(oneline):
        print(f'FAIL: a one-line function that applies no patch was read as an '
              f'uninvoked patcher: {uninvoked_patchers(oneline)}')
        ok = False
    else:
        print('  control: a one-line function that applies no patch does not '
              'inherit the next function\'s body')
    # A harness that defines a patching function and never calls it is the
    # `fpmut.sh` defect, and a runner that delegates is not.
    empty = ("probe(){ python3 -c \"$2\"; }\n")
    runner = ("./lmut.sh\n")
    cov = coverage(['empty_mut.sh', 'run_mut.sh'],
                   read=lambda p: {'empty_mut.sh': empty, 'run_mut.sh': runner}[p])
    # It must NAME the function, not merely answer EMPTY.  The generic
    # `no rows, and it is not a runner` arm answers EMPTY for the same harness,
    # so a control that asked only for the word passed with `uninvoked_patchers`
    # returning nothing -- a diagnosis that cannot be acted on is half a
    # refusal, and naming `probe` is the whole value of that scan.
    if 'probe' not in cov['empty_mut.sh'][0] or \
            not cov['empty_mut.sh'][0].startswith('EMPTY') or \
            not cov['run_mut.sh'][0].startswith('runner'):
        print(f'FAIL: the rowless classification is wrong: {cov}')
        ok = False
    else:
        print('  control: a harness whose patching function is never invoked is '
              'EMPTY and the diagnosis NAMES it; one that delegates to another '
              'harness is a runner')
    return ok


def main():
    ok = _controls()
    applied, bad = scan()
    # FLOOR.  A scan that applied no step reports no no-ops perfectly.
    if applied < 50:
        print(f'FAIL: only {applied} patch steps were applied; the scan is not '
              f'reading the harnesses')
        ok = False
    for h, kind, what, why in bad:
        print(f'FAIL: {h}: a {kind} step [{what}] is DECORATIVE -- '
              f'{why}, so the row runs an unmutated tree')
    cov = coverage()
    # AGREEMENT.  `scan` counts the rows a harness PRINTS when its patches are
    # executed; `coverage` counts the rows its SOURCE declares.  Two derivations
    # from two inputs, so a step the executing side silently stops counting --
    # a baseline row read as a patch, say -- shows up as a disagreement.
    declared = sum(n for _w, n in cov.values())
    if declared != applied:
        print(f'FAIL: the executed scan counted {applied} patch steps and the '
              f'source census declares {declared}; one of them is miscounting')
        ok = False
    for h, (why, _n) in sorted(cov.items()):
        if why.startswith('EMPTY'):
            print(f'FAIL: {h}: {why}; it prints nothing and exits 0, which is '
                  f'indistinguishable from a clean table')
            ok = False
    runners = sorted(h for h, (w, _n) in cov.items() if w.startswith('runner'))
    if not bad and ok:
        print(f'ok: {applied} patch steps across {len(cov) - len(runners)} mutation '
              f'tables, every one of them changes the file it targets; '
              f'{len(runners)} runner(s) carry no rows of their own '
              f'({", ".join(runners)})')
    return 0 if ok and not bad else 1


if __name__ == '__main__':
    sys.exit(main())
