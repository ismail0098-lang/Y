"""Why the validator refuses each kernel that has a loop -- the structural census.

`gap.py` answers "which OPCODES are unmodelled".  It is silent about a whole
second gate, and the two are independent: a kernel with every opcode modelled
can still be refused because its loop STRUCTURE is outside what
`loopval.validate` handles.  Nothing had measured that, so the roadmap ranked
the loop kernels by their opcode gap alone -- which understates them, because
closing the opcode gap would leave them refused for a reason nobody had counted.

WHAT IT REPORTS.  For every corpus kernel with PTX control flow, the refusal
`loopval` gives, aggregated by reason.  `loopval` refuses BY NAME and never
guesses, so its message is a usable census key; that property is what makes this
measurement possible at all.

TWO NORMALISATIONS, and the second one is the point.

  Back-edge counts are folded (`3 back edges` and `5 back edges` are one
  bucket) because the count is a property of the kernel, not of the gap.

  ZERO back edges is kept SEPARATE from more-than-one, although `loopval`
  phrases both as "has N back edges; this validator handles exactly one".
  They are opposite problems -- one is the loop finder coming up empty on a
  kernel that demonstrably branches, the other is capacity -- and the first
  aggregation written here collapsed them and hid four kernels behind
  twenty-seven.  A census key that merges two causes reports the larger one.

THE CENSUS ASKS THE SUITE, NOT ONE MEMBER, and that is what `suite_validate`
is for.  It reported `loopval`'s answer alone, which is a FIRST-REFUSAL reading
of the structural column -- the exact defect `gap.py` exists to avoid on the
opcode column, arrived at here because there used to be only one loop
validator.  `loopval` refuses on the back-edge COUNT before it looks at
anything else, so every kernel with more than one loop was bucketed under a
blocker `nestval` lifts, and MEASURED, at -O3, three genuinely different
blockers were sitting behind it: `gemm_fp8`'s loop carries three branches of
its own, `int8_gemm` branches outside its nest, and `y_cpu_matmul`'s SASS holds
the `@!P0 BRA P1` form -- which is not a new blocker at all, it is one already
counted for `naive_gemm_f32` and `exact_pv`, so its reach was understated.

NOT A VALIDATOR.  A refusal here is the state of the suite today, not a claim
about the kernel.  The counts move when a validator grows, which is the point.
"""
import sys, os, glob, re, collections, time
import loopval
import nestval

BACKEDGE = re.compile(r'(PTX|SASS) has (\d+) back edges')
# `nestval`'s count-based refusals, folded exactly as `BACKEDGE` is.  Left
# unfolded, every distinct count is its own bucket and the census stops
# aggregating -- the same normalisation, one validator over.
TOPLEVEL = re.compile(r'(PTX|SASS) has (\d+) top-level loops')
NOLOOP = re.compile(r'(PTX|SASS) has no loop')
# The ADDRESS and the count in this refusal are properties of the kernel, not
# of the gap -- left in, every kernel is its own singleton bucket and the
# census stops aggregating.  Same normalisation as the back-edge count above.
# `this PTX module holds N entry points (...)` / `this disassembly holds N
# .text sections (...)`.  The parenthesised list is per-kernel; the form is not.
SUBJECT = re.compile(r'this (PTX module|disassembly) holds \d+ (?:entry points|\.text sections)')

UNPLACEABLE = re.compile(r'(PTX|SASS) branch form this CFG cannot place at 0x[0-9a-f]+: (\'[^\']*\')')


def reason_key(msg, kernel=None):
    """Fold a refusal to a census key.  See the docstring for what is kept apart.

    `kernel` NAMES THE SHAPE of a more-than-one-back-edge refusal, and that is
    the third split rather than a nicety.  `loopval` says only how MANY back
    edges it found; the three shapes behind that number need three different
    validators, and the bucket as it stood ranked the cheapest of them first
    while the only kernel a lift is sufficient for sits behind the dearest --
    see `loopcfg.nest_shape`.  The shape comes from `loopcfg`'s own back-edge
    finder, the one `loopval` refuses on, so the two cannot disagree.

    Without a kernel the key degrades to the count-only form.  That is for
    callers that have only a message, and it is deliberately not silent: the
    key then says `shape unknown` rather than naming a shape it did not
    measure."""
    msg = msg.split('\n')[0].strip()
    # A module with no defined subject.  The names of the functions are what
    # varies between kernels and the FORM is what a reader has to act on, so
    # they are folded exactly as the unplaceable-branch form is -- a key that
    # keeps them apart reports one bucket per kernel, which is not a census.
    m = SUBJECT.search(msg)
    if m:
        return (f'{m.group(1)} holds more than one '
                f'{"entry point" if m.group(1) == "PTX module" else ".text section"}')
    m = UNPLACEABLE.search(msg)
    if m:
        # keep the FORM, which is what a lift would have to learn; drop the
        # address and the per-kernel count.
        form = re.sub(r'@!?P\d+\s+', '@P ', m.group(2))
        form = re.sub(r'`\(\.L_\w+\)', '`(.L)', form)
        return f'{m.group(1)} branch form this CFG cannot place: {form}'
    m = NOLOOP.search(msg)
    if m:
        # `nestval`'s phrasing for what `loopval` calls zero back edges.  Kept
        # in that bucket rather than given its own: it is the same fact, and
        # the reason the two are kept apart from the more-than-one bucket is
        # unchanged (one is the loop finder coming up empty, the other is
        # capacity).
        return f'{m.group(1)}: loop finder found NO back edge'
    m = TOPLEVEL.search(msg) or BACKEDGE.search(msg)
    if m:
        side, n = m.group(1), int(m.group(2))
        if n == 0:
            return f'{side}: loop finder found NO back edge'
        shape = 'shape unknown'
        if kernel is not None:
            import loopcfg
            try:
                edges = (loopcfg.ptx_back_edges(f'corpus/{kernel}.ptx')[2] if side == 'PTX'
                         else loopcfg.sass_back_edges(f'corpus/{kernel}.sass')[3])
                kind, cnt, depth = loopcfg.nest_shape(edges)
                shape = kind if kind in ('IRREDUCIBLE',) else f'{kind} depth {depth}'
            except Exception as e:
                shape = f'shape unmeasurable ({str(e)[:40]})'
        # `loopval` says 'back edges' and `nestval` says 'top-level loops'.
        # They are the same fact about the kernel and the key names it
        # once, in the suite's terms rather than in either member's.
        return f'{side}: more than one loop at one level, {shape}'
    return re.sub(r'\s+', ' ', msg)


def has_control_flow(ptxf):
    """A `bra` anywhere in the body, predicated or not.

    Deliberately textual rather than asking `gap.py`: a predicated `bra` whose
    predicate is unrecognised is attributed by that census to the PREDICATE, so
    asking it 'does this kernel have a bra' under-reports.  Found on
    `coprocessor_test`, whose three `bra` are all `@%rt_p0`."""
    return re.search(r'^\s*(@\S+\s+)?bra(\.uni)?\s', open(ptxf).read(), re.M) is not None


def suite_validate(k, budget=20, mode='wide', d='corpus'):
    """The answer the validator SUITE gives, as `(who, verdict, msg, n)`.

    `loopval` handles ONE loop and `nestval` one NEST.  Asking only the first
    reports its first refusal as though it were the suite's, which at -O1 listed
    `y_cpu_matmul` as blocked by a structure `nestval` VALIDATES.

    `nestval` is asked EXACTLY where `loopval`'s refusal is the one it exists to
    lift -- a back-edge COUNT, zero included.  Every other `loopval` refusal
    stands, which is what confines the change to the bucket in question.

    That rule is narrow on purpose and it is not the same as "take the better
    answer".  Measured over the corpus, `loopval` gives a non-count refusal for
    eight kernels.  On four they say the same thing, on one they say the same
    thing in different words (`deterministic_reduce`: 'loop body has more than
    one branch' against 'loop has 2 own branches'), and on THREE they name
    genuinely different blockers (`bn254_msm_bucket`: an unconditional SASS
    back edge against a PTX carry flag inside the nest; the two unsplit
    `paged_decode_attention`: an unplaceable `BRA.DIV` against a PTX branch
    outside the nest).  Both are true and neither is behind the other, so the
    census keeps the member it asked first rather than inventing a metric for
    which refusal is nearer."""
    p, s = f'{d}/{k}.ptx', f'{d}/{k}.sass'
    try:
        v, msg, n = loopval.validate(p, s, budget, mode)
        return 'loopval', v, msg, n
    except Exception as e:
        lmsg = str(e)
    if not BACKEDGE.search(lmsg.split('\n')[0]):
        return 'loopval', 'REFUSED', lmsg, 0
    try:
        v, msg, n = nestval.validate(p, s, budget, mode, verbose=False)
        return 'nestval', v, msg, n
    except Exception as e:
        return 'nestval', 'REFUSED', str(e), 0


def census(kernels, budget=20, mode='wide', d='corpus'):
    rows = []
    for k in kernels:
        t = time.time()
        who, v, msg, n = suite_validate(k, budget, mode, d)
        rows.append((k, v, msg, n, time.time() - t, who))
    return rows


def _selftest():
    """Four controls on the SUITE DISPATCH, three of which perturb the input.

    The census's answer now depends on WHICH member is asked, and that decision
    is invisible in a reason table -- a dispatch that never reaches `nestval`
    reports a perfectly plausible census, which is the shape of every null
    metric in this directory.  So both legs are asserted, in both directions,
    and the `nestval` leg is asserted on a kernel where the two members
    DISAGREE about the verdict rather than merely about the words."""
    import tempfile, shutil
    here = os.path.dirname(os.path.abspath(__file__))
    bad = []

    # 1. THE nestval LEG.  `y_cpu_matmul` at -O1 is refused by `loopval` on a
    #    back-edge count and VALIDATED by `nestval`.  It is the one kernel in
    #    the tree where the two members disagree about the verdict, so it is the
    #    only fixture that can tell a dispatch that reaches `nestval` from one
    #    that does not.  The -O1 build is the committed `o1/` fixture.
    d = tempfile.mkdtemp(prefix='loopgap_selftest_')
    try:
        os.makedirs(os.path.join(d, 'corpus'))
        for ext in ('ptx', 'sass'):
            src = os.path.join(here, 'o1', f'y_cpu_matmul.{ext}')
            if not os.path.exists(src):
                bad.append(f'the -O1 fixture o1/y_cpu_matmul.{ext} is missing; '
                           f'the dispatch control cannot run')
                src = None
            else:
                os.symlink(src, os.path.join(d, 'corpus', f'y_cpu_matmul.{ext}'))
        if not bad:
            who, v, msg, n = suite_validate('y_cpu_matmul',
                                            d=os.path.join(d, 'corpus'))
            if (who, v) != ('nestval', 'VALIDATED'):
                bad.append(f'the suite answers y_cpu_matmul @ -O1 with '
                           f'{who}={v} ({msg.splitlines()[0][:60]}); it should be '
                           f'nestval=VALIDATED -- the nestval leg is not reached')
            else:
                print(f'  control: the suite answers y_cpu_matmul @ -O1 with '
                      f'{who}={v}, {n} obligations -- loopval refuses it on a '
                      f'back-edge count')
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # 2. THE loopval LEG, and it needs a kernel where the two DIFFER or it
    #    asserts nothing.  `bn254_msm_bucket` is refused by `loopval` for a
    #    reason that is not a count (an unconditional SASS back edge) and by
    #    `nestval` for a different one (a PTX carry flag inside the nest), so
    #    which member answered is observable in the message.
    who, v, msg, n = suite_validate('bn254_msm_bucket')
    if who != 'loopval' or 'unconditional' not in msg:
        bad.append(f'bn254_msm_bucket was answered by {who} ({msg.splitlines()[0][:60]}); '
                   f'a non-count loopval refusal must stand')
    else:
        print(f'  control: bn254_msm_bucket keeps loopval\'s non-count refusal')

    # 3. THE FOLD.  Two different counts are one bucket; a count and "no loop"
    #    are NOT, because they are opposite problems (capacity against a loop
    #    finder coming up empty) and the census has kept them apart since it was
    #    written.
    k3 = reason_key('PTX has 3 top-level loops; this validator handles one nest')
    k5 = reason_key('PTX has 5 top-level loops; this validator handles one nest')
    k0 = reason_key('PTX has no loop; use tval.py')
    kb = reason_key('PTX has 3 back edges; this validator handles exactly one')
    if k3 != k5:
        bad.append(f'counts are not folded: {k3!r} against {k5!r}')
    if k3 != kb:
        bad.append(f"loopval's and nestval's phrasings of one fact give two "
                   f"buckets: {kb!r} against {k3!r}")
    if k0 == k3:
        bad.append(f'"no loop" folded into the more-than-one bucket: {k0!r}')
    if not bad:
        print(f'  control: {k3!r} folds both members\' phrasings and every count, '
              f'and is kept apart from {k0!r}')

    # 4. FLOOR.  A control that examined no kernel reports no failures perfectly.
    ks = [os.path.basename(x)[:-4] for x in glob.glob(os.path.join(here, 'corpus', '*.ptx'))]
    if len(ks) < 2:
        bad.append(f'the corpus holds {len(ks)} kernels; there is nothing to dispatch over')

    for b in bad:
        print(f'FAIL: {b}')
    return 1 if bad else 0


if __name__ == '__main__':
    if '--selftest' in sys.argv[1:]:
        sys.exit(_selftest())
    ks = [a for a in sys.argv[1:] if not a.startswith('--')] or sorted(
        os.path.basename(x)[:-4] for x in glob.glob('corpus/*.ptx')
        if has_control_flow(x))
    rows = census(ks)
    # FLOOR.  A census that examined nothing reports "no refusals" perfectly.
    if not rows:
        print('FAIL: examined no kernels -- there is nothing to report'); sys.exit(1)
    agg = collections.Counter(); ex = collections.defaultdict(list)
    nval = 0
    who = collections.Counter()
    for k, v, msg, n, dt, w in rows:
        who[w] += 1
        if v == 'VALIDATED':
            nval += 1; agg['VALIDATED']; ex['VALIDATED'].append(k); agg['VALIDATED'] += 1
        else:
            r = reason_key(msg, k); agg[r] += 1; ex[r].append(k)
    print(f'{len(rows)} kernels with PTX control flow; {nval} validated')
    print(f'   answered by: ' + ', '.join(f'{w} {c}' for w, c in sorted(who.items())) + '\n')
    print(f'{"n":>3}  reason')
    for r, n in agg.most_common():
        print(f'{n:3d}  {r[:92]}')
        print(f'     {", ".join(sorted(ex[r])[:3])}{" ..." if len(ex[r]) > 3 else ""}')
