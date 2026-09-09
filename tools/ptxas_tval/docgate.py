"""The two doc figures that describe a MEASUREMENT, checked against the measurement.

`fpgate.py` already gates one such number (the doc's contraction count against
`contract.py`).  This is the same device applied to the two that had gone stale:
the loop-structure census in `docs/ptxas_translation_validation.md`, and the
tensor-core staging census in `README.md`.

BOTH WERE RIGHT WHEN THEY WERE WRITTEN, and that is what makes a gate the fix
rather than an edit.

  The loop census read `30 PTX: more than one back edge` plus a bucket of `2
  the SASS zero-trip guard is not the last prologue instruction`.  Those two
  kernels are `int8_gemm` and `int8_gemm_scaled`, and an EMITTER change moved
  them: the increment that grid-strided the int8 output tiles in x and y gave
  that kernel two more back edges.  30 + 2 = 32 and the zero-trip bucket is
  empty, so the arithmetic closes exactly.  Nobody editing `ptx_emitter.rs` had
  any reason to re-run a census living in a validator's doc.

  The README's staging census read `89 lines with 2 mma ... 964 / 65 / 22 / 24
  / 7`.  The line count was three increments stale, and two of the five f16
  terms were a raw `grep -c` -- which counts the COMMENT that names the
  instruction.  65 against 64, 22 against 21.  The blockquote four lines above
  that sentence corrects exactly that convention, for a different figure.

WHY THE MEASUREMENT IS NOT RE-IMPLEMENTED HERE.  The census aggregation is
`loopgap`'s own `reason_key`, imported, because a second implementation of it
would agree with the doc while both were wrong -- which is the recorded failure
mode of an agreement gate whose two sides move together.  The instruction count
is the one thing this file does define, and it defines it ONCE, for both rows
and for its own control.

A THIRD FIGURE, ADDED LATER, AND A SECOND FILE.  The doc also publishes that an
OPCODE gap moves with `-O` -- it used to publish the opposite -- and the numbers
behind it are re-derived here by running the real census at both levels through
`frontier.gap_at`, never by re-grepping.  That distinction is not academic: the
figure `int8_gemm_scaled 4 -> 3` was published from a text scan asking only
whether each `-O3` gap opcode still OCCURS at `-O1`, which cannot see a NEW
opcode arriving, and this check caught it on its first run (the answer is
4 -> 4, with `CS2R` arriving).  And the loop census is checked in BOTH files:
the README carried "the largest single lever in the corpus" for an increment
after the doc retracted it, together with its own back-edge total -- a
retraction landing in one file of two, which is the certificate-count defect
one directory over.

THE CONTROLS.  An all-clear is what a gate that measured nothing also reports,
so each check carries a floor AND a positive control that goes through the SAME
code path: a PERTURBED census and a PERTURBED artifact must both be reported.
A control that re-implements the check is a second measurement.
"""
import collections, glob, os, re, sys, tempfile

import frontier
import loopcfg
import loopgap

DOC = '../../docs/ptxas_translation_validation.md'
README = '../../README.md'


def insn_counts(path):
    """Instructions, not lines that mention one.

    THE defect this file exists for.  A PTX comment naming `mma.sync` is not an
    `mma.sync`, and `grep -c` cannot tell them apart."""
    body = [l.strip() for l in open(path) if l.strip() and not l.strip().startswith('//')]
    out = {'lines': sum(1 for _ in open(path))}
    for op in ('mma.sync', 'cp.async', 'ldmatrix', 'bar.sync'):
        out[op] = sum(1 for l in body if op in l)
    return out


def measure_loop_census(only=None):
    """`loopgap`'s aggregation, imported rather than restated.

    `only` restricts the corpus, and exists solely so the measurement's own
    INPUT can be perturbed -- see `the_measurements_read_their_inputs`.  A
    control that perturbs the RESULT cannot tell an independent measurement
    from one that reads the document it is checking.
    """
    ks = sorted(os.path.basename(x)[:-4] for x in glob.glob('corpus/*.ptx')
                if loopgap.has_control_flow(x))
    if only is not None:
        ks = [k for k in ks if k in only]
    agg = collections.Counter()
    nval = 0
    for _k, v, msg, _n, _dt in loopgap.census(ks):
        if v == 'VALIDATED':
            nval += 1
        else:
            agg[loopgap.reason_key(msg)] += 1
    return len(ks), nval, agg


def doc_loop_census(text):
    """The fenced block the doc publishes, as {reason: n}."""
    m = re.search(r'```\n(\d+) kernels with PTX control flow; (\d+) validated\n\n(.*?)```',
                  text, re.S)
    if not m:
        return None
    rows = {}
    for line in m.group(3).splitlines():
        line = line.strip()
        if not line:
            continue
        n, reason = line.split(None, 1)
        rows[reason.strip()] = int(n)
    return int(m.group(1)), int(m.group(2)), rows


def readme_staging_table(text):
    m = re.search(r'```\nkernel\s+lines\s+mma\.sync\s+cp\.async\s+ldmatrix\s+bar\.sync\n(.*?)```',
                  text, re.S)
    if not m:
        return None
    rows = {}
    for line in m.group(1).splitlines():
        if not line.strip():
            continue
        f = line.split()
        rows[f[0]] = dict(zip(('lines', 'mma.sync', 'cp.async', 'ldmatrix', 'bar.sync'),
                              (int(x) for x in f[1:])))
    return rows


def check_loop_census(perturb=None):
    doc = open(DOC).read()
    published = doc_loop_census(doc)
    if published is None:
        print('FAIL: the loop-structure census block is not in the doc at all')
        return 1
    total, nval, rows = published
    n_k, n_v, agg = measure_loop_census()
    if perturb:
        agg = collections.Counter(agg)
        agg[perturb] = agg.get(perturb, 0) + 1
    # FLOOR.  A census that examined nothing agrees with any doc that says so.
    if n_k < 40:
        print(f'FAIL: the census examined {n_k} kernels; it cannot be the corpus')
        return 1
    bad = 0
    if (n_k, n_v) != (total, nval):
        print(f'FAIL: doc says {total} kernels / {nval} validated; measured {n_k} / {n_v}')
        bad += 1
    # Reasons are matched on the doc's own (possibly truncated) text, because
    # the doc wraps and `loopgap` prints a parenthetical the doc drops.
    for reason, n in agg.items():
        hit = [d for d in rows if reason.startswith(d)]
        if not hit:
            print(f'FAIL: measured {n} kernel(s) refusing "{reason}" and the doc has no such row')
            bad += 1
        elif rows[hit[0]] != n:
            print(f'FAIL: doc says {rows[hit[0]]} for "{hit[0]}"; measured {n}')
            bad += 1
    for d, n in rows.items():
        if not any(r.startswith(d) for r in agg):
            print(f'FAIL: the doc publishes a bucket of {n} for "{d}" that nothing measures')
            bad += 1
    # The prose total, DERIVED the way the prose derives it: both sides of the
    # same reason.  It was the half of this figure that stayed right while the
    # block went stale, so it is not redundant with the rows above.
    both = sum(n for r, n in agg.items() if 'more than one back edge' in r)
    m = re.search(r'\*\*(\d+) of (\d+) refuse for one reason: more than one back edge\.\*\*', doc)
    if not m:
        print('FAIL: the prose sentence deriving the back-edge total is gone')
        bad += 1
    elif (int(m.group(1)), int(m.group(2))) != (both, n_k):
        print(f'FAIL: prose says {m.group(1)} of {m.group(2)}; measured {both} of {n_k}')
        bad += 1
    # THE SAME FIGURE IN THE OTHER FILE.  The README carried "the largest single
    # lever in the corpus" for a whole increment after the doc retracted it, and
    # its own back-edge total with it: a retraction that lands in one file of two
    # is the certificate-count defect again, one directory over.  Both files
    # state this census, so both are checked against the one measurement.
    # \s+ rather than a literal space: both files WRAP, so a sentence-shaped
    # pattern with hard spaces in it matches only until someone reflows a
    # paragraph -- and then the gate reports the claim as missing.
    rm = re.search(r'\*\*(\d+)\s+of\s+the\s+(\d+)\s+refuse\s+for\s+one\s+reason:\s+more\s+than\s+one'
                   r'\s+back\s+edge\*\*\s+\((\d+)\s+on\s+the\s+PTX\s+side,\s+(\d+)\s+on\s+the\s+SASS\)',
                   open(README).read())
    if not rm:
        print('FAIL: the README no longer states the back-edge census')
        bad += 1
    else:
        pn = sum(n for r, n in agg.items() if r.startswith('PTX') and 'more than one back edge' in r)
        sn = sum(n for r, n in agg.items() if r.startswith('SASS') and 'more than one back edge' in r)
        got = tuple(int(x) for x in rm.groups())
        if got != (both, n_k, pn, sn):
            print(f'FAIL: README says {got}; measured {(both, n_k, pn, sn)} '
                  '(total, kernels, PTX side, SASS side)')
            bad += 1
    if not bad:
        print(f'ok: loop-structure census, {n_k} kernels, {len(rows)} buckets, '
              f'{both} behind more than one back edge, doc and README agreeing')
    return bad


def check_staging_table(perturb=None):
    rows = readme_staging_table(open(README).read())
    if rows is None:
        print('FAIL: the README staging table is gone')
        return 1
    if len(rows) < 2:
        print(f'FAIL: the staging table has {len(rows)} row(s); it is a COMPARISON')
        return 1
    bad = 0
    for name, published in rows.items():
        path = f'../../tests/{name}'
        if not os.path.exists(path):
            print(f'FAIL: the table names {name}, which is not a committed artifact')
            bad += 1
            continue
        real = insn_counts(path)
        if perturb == name:
            real = dict(real, **{'mma.sync': real['mma.sync'] + 1})
        for k, v in published.items():
            if real[k] != v:
                print(f'FAIL: {name} {k}: README says {v}, the artifact has {real[k]}'
                      + ('  (a raw grep -c counts the comment naming it)'
                         if k != 'lines' and real[k] + 1 == v else ''))
                bad += 1
    if not bad:
        print(f'ok: staging census, {len(rows)} artifacts, counted as instructions')
    return bad


OLEVEL = re.compile(r'`(\w+)`\s*\*\*(\d+)\s*(?:->|\u2192)\s*(\d+)\*\*')


def olevel_pairs(doc):
    """The `kernel **N -> M**` figures out of the -O blockquote.

    Scoped to that blockquote rather than to the whole file, because a
    correcting sentence has to QUOTE the claim it corrects -- the same scoping
    the README layout gate needs, for the same reason."""
    i = doc.find('The second half of this bullet')
    if i < 0:
        return {}
    j = doc.find('\n\n', i)
    return {m.group(1): (int(m.group(2)), int(m.group(3)))
            for m in OLEVEL.finditer(doc[i:j if j > 0 else len(doc)])}


WORDNUM = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}


def stated_back_edges(doc):
    """The back-edge count the doc PUBLISHES for the sufficiency case.

    Parsed rather than hardcoded, because a gate that measures 3 and compares
    it against its own literal 3 leaves the doc's copy of that number gated by
    nothing -- which is what a mutation of the sentence showed: it went stale
    and the whole sweep stayed green.  Same shape as the certificate gate that
    asserted a trust item's ROUTE and never its count."""
    m = re.search(r'multi-back-edge\s+limit,\s+(\w+)\s+on each side', doc)
    if not m:
        return None
    w = m.group(1).lower()
    return WORDNUM.get(w, int(w) if w.isdigit() else None)


def check_optimisation_level_gaps(perturb=None):
    """The doc's claim that an OPCODE gap moves with -O, checked per kernel.

    The published bullet said the opposite, generalised from one kernel where it
    happens to hold.  Each figure here is re-derived
    by running the REAL census twice -- once on the committed corpus, once on
    the same `.ptx` re-assembled at -O1 -- through `frontier.gap_at`, which is
    the call `frontier.py` itself measures with.  A second implementation of
    "build it and ask the census" would agree with the doc while both were
    wrong.

    It also pins the decision-changing half: `y_cpu_matmul`'s -O1 gap must be
    EMPTY on both sides, and its back-edge counts must be the three-on-each-side
    the section states, since that pair is the whole sufficiency case for the
    lift."""
    doc = open(DOC).read()
    pairs = olevel_pairs(doc)
    bad = 0
    # FLOOR.  A parse that recovered nothing agrees with every measurement.
    if len(pairs) < 4:
        print(f'FAIL: recovered {len(pairs)} -O figures from the blockquote; '
              'the parse is not reading it')
        return 1
    for k, (n3, n1) in sorted(pairs.items()):
        g3 = frontier.gap_at(k, None)
        g1 = frontier.gap_at(k, 1)
        m3, m1 = len(g3['sass']), len(g1['sass'])
        if k == perturb:
            m1 += 1
        if (m3, m1) != (n3, n1):
            print(f'FAIL: doc says {k} {n3} -> {n1}; measured {m3} -> {m1} '
                  f'({", ".join(g3["sass"]) or "-"}  ->  {", ".join(g1["sass"]) or "-"})')
            bad += 1
    # the sufficiency case itself
    g = frontier.gap_at('y_cpu_matmul', 1)
    if perturb == 'y_cpu_matmul':
        g['sass'] = g['sass'] + ['SYNTHETIC']
    if g['ptx'] or g['sass']:
        print(f'FAIL: the doc says y_cpu_matmul has an empty opcode gap at -O1; '
              f'measured ptx={g["ptx"]} sass={g["sass"]}')
        bad += 1
    # The back-edge counts, on the -O1 ARTIFACTS -- the doc states them about
    # the -O1 build, and the committed corpus is a different SASS.  Checking
    # the one that happens to be lying around is how a figure ends up true of
    # something other than the thing it is written about.
    with tempfile.TemporaryDirectory() as d:
        s = frontier.build_at('corpus/y_cpu_matmul.ptx', 1, d)
        if s is None:
            print('FAIL: could not assemble y_cpu_matmul at -O1'); return bad + 1
        sp = os.path.join(d, 'y_cpu_matmul.O1.sass')
        open(sp, 'w').write(s)
        n = {}
        for side, fn, path in (('PTX', loopcfg.ptx_regions, 'corpus/y_cpu_matmul.ptx'),
                               ('SASS', loopcfg.sass_regions, sp)):
            try:
                fn(path); n[side] = 1
            except Exception as e:
                mm = re.search(r'has (\d+) back edges', str(e))
                n[side] = int(mm.group(1)) if mm else -1
    want = stated_back_edges(doc)
    # FLOOR.  A sentence this cannot parse must fail loudly: comparing a
    # measurement against `None` would pass silently and gate nothing.
    if want is None:
        print('FAIL: the doc no longer states a back-edge count for the '
              'sufficiency case, so there is nothing to check it against')
        bad += 1
    elif (n['PTX'], n['SASS']) != (want, want):
        print(f'FAIL: the doc says {want} back edges on each side of y_cpu_matmul '
              f'at -O1; measured PTX={n["PTX"]} SASS={n["SASS"]}')
        bad += 1
    if not bad:
        print(f'ok: -O figures, {len(pairs)} kernels re-censused at two levels, '
              'and the y_cpu_matmul sufficiency case')
    return bad


def the_measurements_read_their_inputs():
    """Each measurement must depend on the thing it measures, not on the doc.

    THE HOLE THIS CLOSES.  Both positive controls below used to perturb the
    ANSWER -- `agg` after `measure_loop_census()` returned, `real` after
    `insn_counts()` returned.  That shows the COMPARISON is live and says
    nothing about whether the MEASUREMENT is independent: a measurement
    subverted to read the doc still differs from a perturbed copy of the
    doc's own numbers, so the control reported success while the gate had
    become a self-comparison.  Found by mutation, on a gate written the same
    day.  Perturb the INPUT, through the same call the real measurement uses.
    """
    bad = 0
    # (a) the loop census, restricted to a two-kernel corpus.  A doc-reading
    #     implementation answers with the doc's ~48 whatever it is handed.
    ks = sorted(os.path.basename(x)[:-4] for x in glob.glob('corpus/*.ptx')
                if loopgap.has_control_flow(x))
    if len(ks) < 2:
        print('FAIL: fewer than two control-flow kernels; the input control is vacuous')
        return 1
    sub = ks[:2]
    n_k, _n_v, _agg = measure_loop_census(only=sub)
    if n_k != 2:
        print(f'FAIL: the census was handed {sub} and reported {n_k} kernels; '
              'it is not reading the corpus it was given')
        bad += 1
    # (b) the instruction count, on a copy of a real artifact carrying one
    #     more mma.sync.  A doc-reading implementation cannot move.
    name, published = sorted(readme_staging_table(open(README).read()).items())[0]
    src = f'../../tests/{name}'
    base = insn_counts(src)
    with tempfile.TemporaryDirectory() as d:
        alt = os.path.join(d, name)
        with open(alt, 'w') as f:
            f.write(open(src).read() + '\n\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f0}, {%f1}, {%f2}, {%f3};\n')
        moved = insn_counts(alt)
    if moved['mma.sync'] != base['mma.sync'] + 1:
        print(f'FAIL: an artifact with one extra mma.sync counted '
              f'{moved["mma.sync"]} against {base["mma.sync"]}; '
              'the count is not reading the file it was given')
        bad += 1
    # (c) the -O census, handed a -O3 build in the -O1 slot.  A doc-reading
    #     implementation answers `0` for y_cpu_matmul whatever it assembles.
    g = frontier.gap_at('y_cpu_matmul', 3)
    if not g['sass']:
        print('FAIL: y_cpu_matmul censused at -O3 reported an empty sass gap; '
              'the -O census is not reading the SASS it was handed')
        bad += 1
    if not bad:
        print('  control: each measurement moves when its own INPUT moves, '
              'so none of the three is reading the doc')
    return bad


if __name__ == '__main__':
    bad = check_loop_census() + check_staging_table() + check_optimisation_level_gaps()
    # POSITIVE CONTROLS, through the same code path.  Without these a parse that
    # recovered nothing, or a comparison that compared nothing, reports ok.
    # The FAIL lines they print below are the controls WORKING; they are the
    # gate's own diagnosis of a deliberately perturbed measurement.
    print('\n--- positive controls (the FAIL lines below are EXPECTED) ---')
    if check_loop_census(perturb='PTX: more than one back edge (this validator handles exactly one)') == 0:
        print('FAIL: a perturbed loop census was not reported -- the comparison is dead')
        bad += 1
    if check_staging_table(perturb='int8_gemm.ptx') == 0:
        print('FAIL: a perturbed staging count was not reported -- the comparison is dead')
        bad += 1
    if check_optimisation_level_gaps(perturb='y_cpu_matmul') == 0:
        print('FAIL: a perturbed -O figure was not reported -- the comparison is dead')
        bad += 1
    print('  control: a perturbed census, artifact and -O figure are all reported')
    bad += the_measurements_read_their_inputs()
    if bad:
        print(f'\nFAIL: {bad} doc figure(s) disagree with the measurement.')
        sys.exit(1)
    print('\nok')
