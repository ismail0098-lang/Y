"""Every mutation row's patch must CHANGE the file it targets.

A mutation table is the only evidence in this directory that a check is
load-bearing, and a row whose patch cannot apply runs an UNMUTATED tree and
reports a perfectly plausible line.  That is not hypothetical: `memmut.sh` had
seven such rows when the effect model landed, and this file was written after
`rmut.sh`'s R2 -- "collapse ZERO back edges into the more-than-one bucket" --
turned out to have been decorative since the increment that rewrote
`loopgap.reason_key` underneath it.  Its `s.index` anchor stopped matching, the
heredoc died with `ValueError: substring not found` before it wrote anything,
and the row went on reporting the baseline's numbers.  Nobody re-runs an old
table, so the decay is silent by construction.

WHAT IT CHECKS, and it is the PROPERTY rather than the plumbing.  The obvious
rule is "every patch must assert that it applied", which is a rule about
whether somebody wrote an assertion -- measured when this was written, NINE of
the nine file-writing heredocs in the directory had none.  This applies each
patch step to a COPY of its target and requires the copy to differ.  A row that
asserts nothing is fine as long as its patch bites; a row that asserts and
whose anchor has rotted is caught anyway.

TWO KINDS OF STEP, both executed verbatim rather than parsed.  A `python3 - <<`
heredoc that writes a source file is run with that file's copy as its cwd; a
`sed -i ... FILE.py` line is `eval`ed the same way.  Executing beats matching
because the anchors are shell-quoted regexes inside shell-quoted strings, and a
scanner that re-derives what sed will see gets it wrong in both directions --
the first version of this measurement reported four false stale anchors in
`lmut.sh` for exactly that reason, and running them found all four live.

COMPOUND ROWS.  Some rows apply a second patch on top of a first, so a step can
legitimately be a no-op against a pristine file and bite against the cumulative
one.  A step is live if it changes EITHER, which errs towards silence; the
alternative is an allowlist, which is the defect this directory keeps finding.

CONTROLS.  A floor (a scan that applied no step reports no no-ops perfectly),
and two positive ones that go through the SAME `apply` the real scan uses: a
synthetic harness whose heredoc anchor is absent and one whose sed matches
nothing must both be reported.  A control applied to the RESULT cannot see the
measurement being subverted to read something else.
"""
import glob, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HEREDOC = re.compile(r"python3\s+-\s*<<\s*'?(\w+)'?\n(.*?)\n\1\n", re.S)
WRITES = re.compile(r"open\((['\"])([\w./]+)\1\s*,\s*['\"]w")
SEDLINE = re.compile(r"(?m)^.*\bsed\s+-i\b.*$")
SEDFILE = re.compile(r"([A-Za-z0-9_]+\.py)\s*$")


def steps(harness_src):
    """(kind, target, payload) for every patch step, in file order."""
    out = []
    for m in HEREDOC.finditer(harness_src):
        w = WRITES.search(m.group(2))
        if w:
            out.append(('heredoc', w.group(2), m.group(2), m.start()))
    for m in SEDLINE.finditer(harness_src):
        line = m.group(0)
        f = SEDFILE.search(line)
        if f:
            out.append(('sed', f.group(1), line, m.start()))
    out.sort(key=lambda s: s[3])
    return [(k, t, p) for k, t, p, _ in out]


def apply(kind, target, payload, text):
    """Run one step over `text` in a scratch dir; return the file afterwards."""
    d = tempfile.mkdtemp(prefix='mutgate_')
    try:
        base = os.path.basename(target)
        with open(os.path.join(d, base), 'w') as fh:
            fh.write(text)
        if kind == 'heredoc':
            subprocess.run([sys.executable, '-c', payload], cwd=d,
                           capture_output=True, text=True)
        else:
            subprocess.run(['bash', '-c', payload], cwd=d,
                           capture_output=True, text=True)
        return open(os.path.join(d, base)).read()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def scan(harnesses=None, read=None):
    """(applied, [(harness, kind, target, why)]) for every decorative step."""
    read = read or (lambda p: open(os.path.join(HERE, p)).read())
    bad, applied = [], 0
    for h in harnesses if harnesses is not None else sorted(
            os.path.basename(x) for x in glob.glob(os.path.join(HERE, '*mut*.sh'))):
        try:
            src = read(h)
        except OSError:
            continue
        cum = {}
        for kind, target, payload in steps(src):
            try:
                pristine = read(target)
            except OSError:
                bad.append((h, kind, target, 'target does not exist')); continue
            applied += 1
            after = apply(kind, target, payload, pristine)
            if after != pristine:
                cum[target] = after
                continue
            # a compound step may only bite on top of an earlier one
            base = cum.get(target)
            if base is not None and apply(kind, target, payload, base) != base:
                continue
            bad.append((h, kind, target, 'the patch changes nothing'))
    return applied, bad


def _controls():
    """Two positive controls, through the same `apply` the real scan uses."""
    ok = True
    dead_heredoc = ("s = open('victim.py').read()\n"
                    "s = s.replace('a string that is not in the file', 'x')\n"
                    "open('victim.py', 'w').write(s)\n")
    fake = ("python3 - <<'P'\n" + dead_heredoc + "P\n"
            "sed -i 's/^a line that is not there$/x/' victim.py\n")
    files = {'fake_mut.sh': fake, 'victim.py': 'print(1)\n'}
    n, bad = scan(['fake_mut.sh'], read=lambda p: files[p])
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
    files2 = {'live_mut.sh': live, 'victim.py': 'print(1)\n'}
    n2, bad2 = scan(['live_mut.sh'], read=lambda p: files2[p])
    if n2 != 1 or bad2:
        print(f'FAIL: a step that DOES change its target was reported: {bad2}')
        ok = False
    else:
        print('  control: a step that changes its target is not reported')
    return ok


def main():
    ok = _controls()
    applied, bad = scan()
    # FLOOR.  A scan that applied no step reports no no-ops perfectly.
    if applied < 10:
        print(f'FAIL: only {applied} patch steps were applied; the scan is not '
              f'reading the harnesses')
        ok = False
    for h, kind, target, why in bad:
        print(f'FAIL: {h}: a {kind} step targeting {target} is DECORATIVE -- '
              f'{why}, so the row runs an unmutated tree')
    if not bad:
        print(f'ok: {applied} patch steps across the mutation harnesses, every '
              f'one of them changes the file it targets')
    return 0 if ok and not bad else 1


if __name__ == '__main__':
    sys.exit(main())
