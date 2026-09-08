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

THE CONTROLS.  An all-clear is what a gate that measured nothing also reports,
so each check carries a floor AND a positive control that goes through the SAME
code path: a PERTURBED census and a PERTURBED artifact must both be reported.
A control that re-implements the check is a second measurement.
"""
import collections, glob, os, re, sys, tempfile

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
    if not bad:
        print(f'ok: loop-structure census, {n_k} kernels, {len(rows)} buckets, '
              f'{both} behind more than one back edge')
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
    if not bad:
        print('  control: each measurement moves when its own INPUT moves, '
              'so neither is reading the doc')
    return bad


if __name__ == '__main__':
    bad = check_loop_census() + check_staging_table()
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
    print('  control: a perturbed census and a perturbed artifact are both reported')
    bad += the_measurements_read_their_inputs()
    if bad:
        print(f'\nFAIL: {bad} doc figure(s) disagree with the measurement.')
        sys.exit(1)
    print('\nok')
