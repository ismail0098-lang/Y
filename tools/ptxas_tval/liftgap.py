"""What the multi-back-edge lift would leave behind -- the SECOND refusal.

`loopgap.py` reports the FIRST structural refusal and stops.  That is the
first-refusal problem `gap.py` exists to solve, one layer over, and the doc
already names it about this very lift: "lifting the PTX back-edge check can
expose another; what `loopval` would say after that is not measurable without
building it."

IT IS MEASURABLE WITHOUT BUILDING IT.  `loopcfg` refuses on the back-edge count
BEFORE it looks at anything else, so the checks behind that one have simply
never been asked.  Ask them: decompose the nest, and run the remaining
structural predicates at every level the lift would produce.

WHAT A LIFT WOULD HAVE TO DO, and why the levels are what to ask about.  An
inner loop cannot be executed straight-line, so a nested validator summarises it
by its own proved relation and inducts over the nest.  Each level is then
validated as a loop in its own right, with its children summarised -- so each
level's OWN body (its region minus its immediate children's regions) has to
satisfy the same predicates `loopval` applies to a single loop's body today.

WHAT IS REPORTED, per level, and it is deliberately only the NAMED checks:

  store-in-body     `loopval` compares the stores AFTER the loop, so a store in
                    the body is refused.  A store in an outer level's body is
                    the ordinary shape of a tiled kernel and it is the thing
                    this census exists to surface.
  branch-in-body    a branch that is not the level's own exit test and not a
                    child loop's structure.

WHAT IT DOES NOT AND CANNOT SAY.  It never reports that a kernel WOULD validate.
Past these predicates lie the opcode gap (censused separately), the relation
proposal, and the obligations themselves -- none of which is decidable by
reading.  A level this file reports as clear is a level with no NAMED structural
refusal, and nothing more.

THE STORE SCAN IS TEXTUAL, and that is sound in the direction that matters.
`ptxexec` appends to `stores` exactly on `st.global.*`, and `sassexec` on
`STG.*`, so matching those prefixes is the same set for every shape the
executors model.  A body carrying an unmodelled opcode is refused by the OPCODE
census anyway, so a disagreement there cannot make this report optimistic.
"""
import glob, os, re, sys

import loopcfg

# The store opcodes the two executors actually record -- see the docstring.
#
# THE PREDICATE PREFIX IS PART OF THE INSTRUCTION, and leaving it out of these
# patterns made this census OPTIMISTIC -- the one direction the docstring above
# claims it cannot be.  `y_cpu_matmul`'s store is `@%p11 st.global.f32`, so an
# anchored `^st\.global` matched none of the stores that matter and the kernel
# came back with no store-in-body refusal at all.  Caught by checking the report
# against a kernel whose store had already been read by eye.
PTX_STORE  = re.compile(r'^(?:@!?%\w+\s+)?st\.global\b')
SASS_STORE = re.compile(r'^(?:@!?P\d+\s+)?STG\b')
PTX_BRA    = loopcfg.PTX_BRA
SASS_BRA   = loopcfg.SASS_BRA


def _children(iv, i):
    """Indices of the back edges IMMEDIATELY inside back edge `i`."""
    h, e = iv[i][0], iv[i][1]
    inside = [j for j in range(len(iv))
              if j != i and h <= iv[j][0] and iv[j][1] <= e]
    return [j for j in inside
            if not any(k != j and iv[k][0] <= iv[j][0] and iv[j][1] <= iv[k][1]
                       for k in inside)]


def _own(items, key, lo, hi, holes):
    """`items` between lo and hi, minus the closed ranges in `holes`."""
    out = []
    for it in items:
        k = key(it)
        if not (lo < k < hi):
            continue
        if any(a <= k <= b for a, b in holes):
            continue
        out.append(it)
    return out


def levels(ptx_path, sass_path):
    """Per side, per loop level: the level's own body and what refuses it."""
    raw, _lab, pbacks = loopcfg.ptx_back_edges(ptx_path)
    ins, _l2, _trap, sbacks = loopcfg.sass_back_edges(sass_path)

    def side(backs, items, key, text, store_re, bra_re):
        iv = [(b[0], b[1]) for b in backs]
        out = []
        for i, (h, e) in enumerate(iv):
            holes = [(iv[j][0], iv[j][1]) for j in _children(iv, i)]
            body = _own(items, key, h, e, holes)
            txt = [text(x) for x in body]
            stores = [t for t in txt if store_re.match(t)]
            # the level's own exit test is a branch and is expected; anything
            # beyond one is the shape `loopcfg` already refuses for a flat loop.
            bras = [t for t in txt if bra_re.fullmatch(t)]
            out.append({'span': (h, e), 'n': len(txt),
                        'stores': len(stores), 'bras': len(bras)})
        return out

    p = side(pbacks, [(i, t) for i, (k, t) in enumerate(raw) if k == 'i'],
             lambda x: x[0], lambda x: x[1], PTX_STORE, PTX_BRA)
    s = side(sbacks, ins, lambda x: x[0], lambda x: x[1], SASS_STORE, SASS_BRA)
    return p, s


def report(kernel, d='corpus'):
    p, s = levels(f'{d}/{kernel}.ptx', f'{d}/{kernel}.sass')
    pshape = loopcfg.nest_shape(loopcfg.ptx_back_edges(f'{d}/{kernel}.ptx')[2])
    blockers = []
    ps = sum(1 for L in p if L['stores'])
    ss = sum(1 for L in s if L['stores'])
    if ps: blockers.append(f'store in the body of {ps} PTX level(s)')
    if ss: blockers.append(f'store in the body of {ss} SASS level(s)')
    # a level whose own body still branches, over and above its exit test
    pb = sum(1 for L in p if L['bras'] > 1)
    sb = sum(1 for L in s if L['bras'] > 0)
    if pb: blockers.append(f'{pb} PTX level(s) branch beyond their exit test')
    if sb: blockers.append(f'{sb} SASS level(s) branch inside the body')
    return pshape, p, s, blockers


def selftest():
    """Controls, and all of them perturb the INPUT.

    An all-clear is what a census that decomposed nothing also reports, and a
    store scan that matched nothing reports "no store-in-body" perfectly -- which
    is exactly what the first version of this file did, because it anchored on
    `^st.global` and every store that matters is PREDICATED.  So the controls
    feed real artifacts through the same call and require the answer to move.
    """
    import tempfile, shutil
    bad = 0

    # (a) A PREDICATED store must be seen.  This is the bug this file shipped
    #     with for one run: the scan missed every store and reported 23 of 32
    #     kernels as left with nothing, the opposite of the truth.
    for pat, sample in ((PTX_STORE,  '@%p11 st.global.f32 [%rd11], %f0'),
                        (SASS_STORE, '@P0 STG.E desc[UR4][R2.64], R7')):
        if not pat.match(sample):
            print(f'FAIL: {sample!r} is not recognised as a store; the scan is '
                  'blind to predicated stores')
            bad += 1
    # ... and a non-store must NOT be, or the scan is a constant.
    for pat, sample in ((PTX_STORE, 'ld.global.f32 %f1, [%rd5]'),
                        (SASS_STORE, 'LDG.E R7, desc[UR4][R2.64]')):
        if pat.match(sample):
            print(f'FAIL: {sample!r} is counted as a store; the scan reports one '
                  'for everything')
            bad += 1

    # (b) PERTURB THE ARTIFACT.  Take a kernel this census reports as having a
    #     store in one PTX level, delete that store, and the count must fall.
    k = 'y_cpu_matmul'
    base = report(k)
    nb = sum(1 for L in base[1] if L['stores'])
    if nb == 0:
        print(f'FAIL: the control is stated over {k}, which no longer has a '
              'store in any PTX level')
        bad += 1
    else:
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(f'corpus/{k}.sass', f'{d}/{k}.sass')
            out = [l for l in open(f'corpus/{k}.ptx')
                   if not PTX_STORE.match(l.strip().rstrip(';'))]
            open(f'{d}/{k}.ptx', 'w').writelines(out)
            moved = report(k, d)
        nm = sum(1 for L in moved[1] if L['stores'])
        if nm >= nb:
            print(f'FAIL: {k} with its stores deleted still reports {nm} PTX '
                  f'level(s) with a store, against {nb}; the census is not '
                  'reading the file it was given')
            bad += 1

    # (c) NON-VACUITY: some level must have a store and some must not, or the
    #     per-level split distinguishes nothing whatever the scan says.
    with_, without = 0, 0
    for x in glob.glob('corpus/*.ptx'):
        kk = os.path.basename(x)[:-4]
        if not os.path.exists(f'corpus/{kk}.sass'):
            continue
        try: _sh, pl, _sl, _bl = report(kk)
        except Exception: continue
        with_   += sum(1 for L in pl if L['stores'])
        without += sum(1 for L in pl if not L['stores'])
    if not with_ or not without:
        print(f'FAIL: {with_} levels with a store and {without} without; '
              'the per-level split is vacuous')
        bad += 1

    # (d) THE CENSUS AND ITS FLOOR, through the same call the CLI uses.  A
    #     mutation neutering the loop used to leave every gate green, because
    #     nothing but `__main__` ran it.  Handed a corpus of one single-loop
    #     kernel it must decompose nothing and RAISE.
    try:
        _rows, ex, lv = census()
        if ex < 2 or lv < ex:
            print(f'FAIL: the census reports {ex} kernels / {lv} levels'); bad += 1
    except Exception as e:
        print(f'FAIL: the census raised on the real corpus: {e}'); bad += 1
    single = [k for k in sorted(os.path.basename(x)[:-4] for x in glob.glob('corpus/*.ptx'))
              if os.path.exists(f'corpus/{k}.sass')
              and loopcfg.nest_shape(loopcfg.ptx_back_edges(f'corpus/{k}.ptx')[2])[1] <= 1][:1]
    if not single:
        print('FAIL: no single-loop kernel to probe the floor with'); bad += 1
    else:
        try:
            census(single)
            print(f'FAIL: the census accepted a corpus of {single} and reported a '
                  'result; the floor does not fire'); bad += 1
        except Exception:
            pass
    if not bad:
        print(f'  control: predicated stores are seen, the count falls when the '
              f'artifact loses them, the corpus really holds {with_} levels '
              f'with a store and {without} without, and the floor fires on an '
              f'empty census')
    return bad


def census(ks=None):
    """The whole second-refusal census, and its FLOOR.

    Extracted from `__main__` because a floor that only runs when a human types
    the command is guarded by nothing: a mutation neutering the loop left every
    gate green, since `--selftest` returns before reaching it and the doc gate
    calls `report` directly.  Both consumers run this now.

    Raises rather than exiting, so a caller can control a deliberate emptiness."""
    if ks is None:
        ks = sorted(os.path.basename(x)[:-4] for x in glob.glob('corpus/*.ptx')
                    if os.path.exists(x[:-4] + '.sass'))
    rows, examined, nlevels = [], 0, 0
    for k in ks:
        try:
            shape, p, s, blockers = report(k)
        except Exception:
            continue
        if shape[1] <= 1:
            continue                      # the lift is not what this kernel needs
        examined += 1; nlevels += len(p)
        rows.append((k, shape, len(p), len(s), blockers))
    # FLOOR.  A census that decomposed nothing reports "no second refusal"
    # perfectly -- the null metric this repository keeps meeting.
    if not examined or not nlevels:
        raise Exception(f'decomposed {examined} kernels / {nlevels} levels -- '
                        'there is nothing to report')
    return rows, examined, nlevels


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(1 if selftest() else 0)
    try:
        rows, examined, nlevels = census(sys.argv[1:] or None)
    except Exception as e:
        print(f'FAIL: {e}'); sys.exit(1)
    print(f'{examined} multi-back-edge kernels, {nlevels} PTX loop levels\n')
    print(f'{"kernel":34s} {"shape":<16s} lv  what the lift would STILL leave')
    clear = []
    for k, sh, np_, ns, bl in sorted(rows, key=lambda r: (len(r[4]), r[0])):
        print(f'{k:34s} {sh[0]+" d"+str(sh[2]):<16s} {np_:2d}  '
              f'{"; ".join(bl) if bl else "-- no NAMED structural refusal left"}')
        if not bl: clear.append(k)
    print(f'\n{len(clear)} of {examined} would be left with no named structural '
          f'refusal: {", ".join(clear) if clear else "(none)"}')
