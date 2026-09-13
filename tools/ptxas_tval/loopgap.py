"""Why `loopval` refuses each kernel that has a loop -- the structural census.

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

NOT A VALIDATOR.  A refusal here is the state of `loopval` today, not a claim
about the kernel.  The counts move when `loopval` grows, which is the point.
"""
import sys, os, glob, re, collections, time
import loopval

BACKEDGE = re.compile(r'(PTX|SASS) has (\d+) back edges')
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
    m = BACKEDGE.search(msg)
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
        return f'{side}: more than one back edge, {shape}'
    return re.sub(r'\s+', ' ', msg)


def has_control_flow(ptxf):
    """A `bra` anywhere in the body, predicated or not.

    Deliberately textual rather than asking `gap.py`: a predicated `bra` whose
    predicate is unrecognised is attributed by that census to the PREDICATE, so
    asking it 'does this kernel have a bra' under-reports.  Found on
    `coprocessor_test`, whose three `bra` are all `@%rt_p0`."""
    return re.search(r'^\s*(@\S+\s+)?bra(\.uni)?\s', open(ptxf).read(), re.M) is not None


def census(kernels, budget=20, mode='wide'):
    rows = []
    for k in kernels:
        p, s = f'corpus/{k}.ptx', f'corpus/{k}.sass'
        t = time.time()
        try:
            v, msg, n = loopval.validate(p, s, budget, mode)
        except Exception as e:
            v, msg, n = 'REFUSED', str(e), 0
        rows.append((k, v, msg, n, time.time() - t))
    return rows


if __name__ == '__main__':
    ks = sys.argv[1:] or sorted(
        os.path.basename(x)[:-4] for x in glob.glob('corpus/*.ptx')
        if has_control_flow(x))
    rows = census(ks)
    # FLOOR.  A census that examined nothing reports "no refusals" perfectly.
    if not rows:
        print('FAIL: examined no kernels -- there is nothing to report'); sys.exit(1)
    agg = collections.Counter(); ex = collections.defaultdict(list)
    nval = 0
    for k, v, msg, n, dt in rows:
        if v == 'VALIDATED':
            nval += 1; agg['VALIDATED']; ex['VALIDATED'].append(k); agg['VALIDATED'] += 1
        else:
            r = reason_key(msg, k); agg[r] += 1; ex[r].append(k)
    print(f'{len(rows)} kernels with PTX control flow; {nval} validated\n')
    print(f'{"n":>3}  reason')
    for r, n in agg.most_common():
        print(f'{n:3d}  {r[:92]}')
        print(f'     {", ".join(sorted(ex[r])[:3])}{" ..." if len(ex[r]) > 3 else ""}')
