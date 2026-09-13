"""Split a PTX kernel and the SASS it became into (prologue, body, epilogue)
around ONE natural loop.

Everything here is fail-closed by name.  A shape this does not recognise is a
refusal, not a best guess: a validator that mis-identifies a loop proves a
theorem about a program nobody wrote.

Recognised shape, and nothing else:

    prologue                       may branch FORWARD to the loop exit only
                                   (the zero-trip guard ptxas emits)
  header:                          <- the back edge's target
    body                           may not branch out
    <cond> goto header             <- exactly one back edge
  exit:
    epilogue

The back edge's guard is the loop-continue condition.  The PTX writes it as a
guard at the TOP (`setp.ge; @p bra END`) and ptxas rewrites it as a test at the
BOTTOM, so the two are not the same expression and are related, not compared
syntactically.
"""
import re

# ---------------- loop-nest SHAPE ----------------
#
# WHY THIS IS NOT A DETAIL OF THE REFUSAL MESSAGE.  `loopval` handles exactly
# one back edge, and the census that reports why folds every other count into a
# single bucket -- "more than one back edge".  That bucket holds THREE shapes
# which need three different validators, and merging them ranks the cheapest
# lift first while the only kernel a lift is SUFFICIENT for sits behind the
# dearest one:
#
#   SEQUENTIAL  one loop after another.  The same relation, proved once per
#               loop and composed at the join.  The cheap lift.
#   NESTED      one loop inside another.  An inner loop cannot be executed
#               straight-line, so it has to be SUMMARISED by its own proved
#               relation and the induction runs over the nest.  The dear lift.
#   MIXED       both, so it needs both.
#
# `loopgap.py`'s own docstring already records the general form of this -- "a
# census key that merges two causes reports the larger one" -- for the split
# between zero back edges and more than one.  This is the same observation one
# level in, on the bucket that split left behind.
#
# IRREDUCIBLE is a refusal, not a fourth shape: two back edges that neither
# nest nor sit apart share a header region no structured lift describes.

def nest_shape(backs):
    """(kind, count, depth) for a list of (header, edge, ...) back edges.

    Positions are compared as intervals: `[h, e]` contains `[h2, e2]` iff
    `h <= h2 and e2 <= e`.  Indices for PTX, addresses for SASS -- both are
    monotone in program order, which is all this needs."""
    iv = [(b[0], b[1]) for b in backs]
    n = len(iv)
    if n == 0: return 'NONE', 0, 0
    if n == 1: return 'SINGLE', 1, 1
    nest = seq = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = iv[i], iv[j]
            if (a[0] <= b[0] and b[1] <= a[1]) or (b[0] <= a[0] and a[1] <= b[1]):
                nest += 1
            elif a[1] < b[0] or b[1] < a[0]:
                seq += 1
            else:
                return 'IRREDUCIBLE', n, 0
    depth = max(1 + sum(1 for j in range(n)
                        if j != i and iv[j][0] <= iv[i][0] and iv[i][1] <= iv[j][1])
                for i in range(n))
    if nest and seq: return 'MIXED', n, depth
    if nest:         return 'NESTED', n, depth
    return 'SEQUENTIAL', n, depth


def shape_of(ptx_path, sass_path):
    """The shape both sides present, as ((kind,n,d), (kind,n,d))."""
    return (nest_shape(ptx_back_edges(ptx_path)[2]),
            nest_shape(sass_back_edges(sass_path)[3]))


# ---------------- PTX ----------------
PTX_LABEL = re.compile(r'^\$?([A-Za-z_][\w$]*)\s*:$')

# The predicate is a NAME, not a number.  It was `%p(\d+)` and that is a
# lexical assumption about the emitter's register naming, not a fact about PTX:
# `ptx_emitter` writes `%rt_p0` and `%qp0` in the coprocessor kernels, and a
# branch guarded by one of those did not match, so it was NOT A BRANCH to this
# CFG.  Six corpus kernels have two back edges each that were invisible for
# exactly that reason, and the census called them "loop finder found NO back
# edge" -- a misdiagnosis, in the fail-OPEN direction, in a CFG.
PTX_BRA   = re.compile(r'^(?:@(!?)%([\w$]+)\s+)?bra(?:\.uni)?\s+\$?([\w$]+)$')

# The branch FAMILY, for the same reason `SASS_BRANCHY` exists on the other
# side: widening one pattern fixes one instance, whereas refusing anything in
# the family this CFG cannot place catches the next one.  The PTX side never
# had this and the SASS side has had it since the branch-form work.
PTX_BRANCHY = re.compile(r'^(?:@!?%[\w$]+\s+)?(bra|brx)\b')

# A module's ENTRY POINTS.  `.entry` may carry `.visible` or `.weak`.
PTX_ENTRY = re.compile(r'^\s*(?:\.(?:visible|weak|extern)\s+)*\.entry\s+([\w$]+)', re.M)


def ptx_entry_points(path):
    """Every `.entry` in the module, in file order."""
    return PTX_ENTRY.findall(open(path).read())


def ptx_unclassified_branches(raw):
    """Branch-family instructions `PTX_BRA` cannot parse, as (index, text).

    Returned rather than raised, so the SHAPE census keeps its resolution; the
    validator path refuses on them."""
    return [(i, t) for i, (k, t) in enumerate(raw)
            if k == 'i' and PTX_BRANCHY.match(t) and not PTX_BRA.fullmatch(t)]


def ptx_pred_index(name):
    """The `%pN` index of a guard predicate, or None if it is not of that form.

    `ptx_regions` hands this to `ptxexec`, whose predicate file is keyed by
    NUMBER.  Widening `PTX_BRA` to see a `%rt_p0` branch therefore makes the
    branch visible without making it executable, and the honest report is a
    refusal naming the name-keyed register file -- not `int('rt_p0')`, which is
    the `invalid literal for int()` crash this directory already records."""
    m = re.fullmatch(r'p(\d+)', name or '')
    return int(m.group(1)) if m else None

def ptx_back_edges(path):
    """Every PTX back edge, as (header_index, edge_index, match).

    Split out of `ptx_regions` so that the SHAPE census and this validator's own
    arity check read ONE back-edge finder.  A second implementation of it would
    agree with this one while both were wrong -- the recorded failure mode of an
    agreement gate whose two sides move together -- and the shape of a nest is
    exactly what decides which validator a kernel needs."""
    ents = ptx_entry_points(path)
    if len(ents) > 1:
        raise Exception(
            f'this PTX module holds {len(ents)} entry points ({", ".join(ents)}); '
            f'which one is under test is undefined  (refusing, not guessing)')
    raw = []
    depth = 0
    for line in open(path):
        s = line.strip()
        if s.startswith('//') or not s: continue
        if s == '{': depth += 1; continue
        if s == '}': depth -= 1; continue
        if not depth: continue
        if s.startswith('.'): continue          # .reg / .maxnreg / .loc are not instructions
        if s.endswith(';'): raw.append(('i', s[:-1].strip()))
        elif PTX_LABEL.match(s): raw.append(('l', PTX_LABEL.match(s).group(1)))
    lab = {t: i for i, (k, t) in enumerate(raw) if k == 'l'}
    backs = []
    for i, (k, t) in enumerate(raw):
        if k != 'i': continue
        m = PTX_BRA.fullmatch(t)
        if m and lab.get(m.group(3).lstrip('$'), 1 << 30) < i:
            backs.append((lab[m.group(3).lstrip('$')], i, m))
    return raw, lab, backs


def ptx_regions(path):
    raw, lab, backs = ptx_back_edges(path)
    unk = ptx_unclassified_branches(raw)
    if unk:
        i, t = unk[0]
        raise Exception(f'PTX branch form this CFG cannot place: {t!r} '
                        f'({len(unk)} in this kernel)  (refusing, not guessing)')
    if len(backs) != 1:
        raise Exception(f'PTX has {len(backs)} back edges; this validator handles '
                        f'exactly one  (refusing, not guessing)')
    h, e, m = backs[0]
    if m.group(2) is not None:
        raise Exception('PTX back edge is predicated; the recognised shape tests at '
                        'the TOP of the body  (refusing, not guessing)')
    # the top guard: the first predicated forward `bra` after the header
    guard = None
    for i in range(h + 1, e):
        k, t = raw[i]
        if k != 'i': continue
        mm = PTX_BRA.fullmatch(t)
        if mm:
            if guard is not None:
                raise Exception('PTX loop body has more than one branch  '
                                '(refusing, not guessing)')
            tgt = lab.get(mm.group(3).lstrip('$'))
            if tgt is None or tgt <= e:
                raise Exception('PTX loop body branches somewhere other than the loop '
                                'exit  (refusing, not guessing)')
            guard = (i, mm, tgt)
    if guard is None:
        raise Exception('no exit test found in the PTX loop body  (refusing, not guessing)')
    gi, gm, gexit = guard
    gp = ptx_pred_index(gm.group(2))
    if gp is None:
        raise Exception(
            f'PTX exit test is guarded by %{gm.group(2)}, and this validator\'s '
            f'predicate file is keyed by number; a named predicate register needs '
            f'the name-keyed refactor  (refusing, not guessing)')
    ins = lambda a, b: [t for k, t in raw[a:b] if k == 'i']
    return {
        'prologue': ins(0, h),
        'guard':    raw[gi][1],                    # the `@%pN bra EXIT` line
        'guard_pred': (gm.group(1) == '!', gp),
        'pre_guard': ins(h + 1, gi),               # setp etc. before the test
        'body':     ins(gi + 1, e),                # excludes the back edge
        'epilogue': ins(gexit, len(raw)),
    }

# ---------------- SASS ----------------
SASS_INSN = re.compile(r'^\s*/\*([0-9a-f]+)\*/\s+(.*?);\s*$')
SASS_LBL  = re.compile(r'^(\.L_\w+):')
SASS_BRA  = re.compile(r'^(?:@(!?)P(\d+)\s+)?BRA\s+`\((\.L_\w+)\)$')

# A disassembly's per-function `.text.` sections.  Each one RESTARTS addressing
# at /*0000*/, so two of them in one file put two functions in ONE address
# space: `lab` maps a label to an address that names two instructions, and the
# `target <= addr` back-edge test compares addresses across functions.  Measured
# on the corpus: the two split paged-decode kernels carry 376 and 248 duplicated
# addresses, and every back edge reported for them is an artifact of that.
SASS_SECTION = re.compile(r'^\s*\.text\.([\w$.]+):', re.M)


def sass_sections(path):
    """Every `.text.<name>` section in the disassembly, in file order."""
    return SASS_SECTION.findall(open(path).read())


def sass_back_edges(path):
    """Every SASS back edge, as (header_addr, edge_addr, match).

    See `ptx_back_edges` for why this is factored out rather than duplicated."""
    secs = sass_sections(path)
    if len(secs) > 1:
        raise Exception(
            f'this disassembly holds {len(secs)} .text sections ({", ".join(secs)}); '
            f'each restarts addressing at 0, so their labels and addresses collide '
            f'(refusing, not guessing)')
    text = open(path).read()
    lab = {m.group(1): int(m.group(2), 16)
           for m in re.finditer(r'(\.L_\w+):\s*\n\s*/\*([0-9a-f]+)\*/', text)}
    ins = []
    for line in text.splitlines():
        m = SASS_INSN.match(line)
        if m: ins.append((int(m.group(1), 16), m.group(2).strip()))
    # nvcc's trailing `.L_x: BRA .L_x` self-loop after EXIT is dead code
    trap = {a for a, t in ins
            if (m := SASS_BRA.fullmatch(t)) and lab.get(m.group(3)) == a}
    backs = []
    for a, t in ins:
        if a in trap: continue
        m = SASS_BRA.fullmatch(t)
        if m and m.group(3) in lab and lab[m.group(3)] <= a:
            backs.append((lab[m.group(3)], a, m))
    return ins, lab, trap, backs


# A branch-family mnemonic.  `SASS_BRA` recognises ONE form; anything else in
# this family is a control transfer the CFG cannot place, and the fail-open
# reading of that -- "it did not match, so it is not a branch" -- is what makes
# it dangerous.  A backward one would be a loop invisible to `sass_regions`,
# which would then hand `loopval` a "body" that actually loops.
#
# Deliberately NOT in this family: `BSSY`/`BSYNC` (reconvergence bookkeeping,
# they transfer no control), `WARPSYNC`, `EXIT`, `RET`.  Including those would
# refuse almost every kernel with control flow and destroy the census rather
# than sharpen it.  `CALL` is a genuine transfer and is left to `sassexec`,
# which refuses it by name.
SASS_BRANCHY = re.compile(r'^(?:@!?P\d+\s+)?(BRA|BRX|JMP|JMX)\b')


def sass_unclassified_branches(ins):
    """Branch-family instructions `SASS_BRA` cannot parse, as (addr, text).

    Returned rather than raised so the SHAPE census keeps its resolution: a
    kernel that also has, say, an irreducible back-edge pair should be able to
    report that.  `sass_regions` -- the validator path -- refuses on this."""
    return [(a, t) for a, t in ins
            if SASS_BRANCHY.match(t) and not SASS_BRA.fullmatch(t)]


def sass_regions(path):
    ins, lab, trap, backs = sass_back_edges(path)
    # REFUSE a branch form the CFG cannot place, before anything is concluded
    # from the back-edge count -- the count is only meaningful if every branch
    # was seen.  Latent when this was written: all 13 in the corpus are forward
    # and every kernel holding one is refused earlier for an opcode.
    unk = sass_unclassified_branches(ins)
    if unk:
        a, t = unk[0]
        raise Exception(f'SASS branch form this CFG cannot place at 0x{a:x}: {t!r} '
                        f'({len(unk)} in this kernel)  (refusing, not guessing)')
    if len(backs) != 1:
        raise Exception(f'SASS has {len(backs)} back edges; this validator handles '
                        f'exactly one  (refusing, not guessing)')
    h, e, m = backs[0]
    if m.group(2) is None:
        raise Exception('SASS back edge is unconditional  (refusing, not guessing)')
    body_ins = [(a, t) for a, t in ins if h <= a < e]
    exit_addr = min((a for a, _ in ins if a > e), default=None)
    # nothing in the body may branch
    for a, t in body_ins:
        if SASS_BRA.fullmatch(t):
            raise Exception(f'SASS loop body branches at 0x{a:x}  (refusing, not guessing)')
    # the prologue may branch only to the loop exit (the zero-trip guard)
    for a, t in ins:
        if a >= h or a in trap: continue
        mm = SASS_BRA.fullmatch(t)
        if mm and lab.get(mm.group(3)) != exit_addr:
            raise Exception(f'SASS prologue branches to {mm.group(3)} rather than the '
                            f'loop exit  (refusing, not guessing)')
    return {
        'prologue': [(a, t) for a, t in ins if a < h],
        'body':     body_ins,
        'back':     (e, m),
        'epilogue': [(a, t) for a, t in ins if a > e and a not in trap],
        'labels':   lab, 'exit_addr': exit_addr,
    }


def _self_check():
    r"""Pin the branch patterns AT IMPORT, because behaviour cannot reach them.

    Widening `PTX_BRA` from `%p(\d+)` to a name changed NO standing result --
    every kernel it newly sees is refused for another reason -- so there is no
    behavioural gate that can fail if it is reverted, and the `fpmode`/`sassexec`
    remedy applies: a tool that imports this dies rather than serving a CFG that
    cannot see a branch.  Both directions are pinned, because an over-refusal
    here is as bad as an under-refusal: the recorded instance of the second is
    six kernels whose loops were invisible, and of the first is a strict parse
    that moved nine standing results in an afternoon."""
    # the NAMED forms first: reverting the widening then fails with "refuses the
    # legitimate form '@%rt_p0 ...'", which names the defect, rather than with a
    # remark about how `%p3`'s group is spelled.
    ok = [('@%rt_p0 bra $RT_KNN_0',    'rt_p0'),
          ('@%qp0 bra $Q_0',           'qp0'),
          ('bra $L_0',                 None),
          ('bra.uni $L_0',             None),
          ('@%p3 bra $L_0',            'p3'),
          ('@!%p12 bra L_0',           'p12')]
    for t, pred in ok:
        m = PTX_BRA.fullmatch(t)
        if not m:
            raise Exception(f'loopcfg: PTX_BRA refuses the legitimate form {t!r}  '
                            '(over-refusal: a branch this CFG cannot see is a loop '
                            'it cannot see)')
        if m.group(2) != pred:
            raise Exception(f'loopcfg: PTX_BRA reads the predicate of {t!r} as '
                            f'{m.group(2)!r}, not {pred!r}')
    for t in ('ld.global.f32 %f1, [%rd2]', 'setp.ge.s32 %p1, %r1, %r2', 'bar.sync 0'):
        if PTX_BRA.fullmatch(t):
            raise Exception(f'loopcfg: PTX_BRA accepts the non-branch {t!r}')
    # the family must be WIDER than the form, or nothing is ever unplaceable
    if not PTX_BRANCHY.match('@%p0 brx.idx %r1, $T'):
        raise Exception('loopcfg: PTX_BRANCHY does not cover `brx`, so an indirect '
                        'branch is invisible rather than refused')
    if PTX_BRA.fullmatch('@%p0 brx.idx %r1, $T'):
        raise Exception('loopcfg: PTX_BRA claims to parse `brx`')
    if PTX_BRANCHY.match('ld.global.f32 %f1, [%rd2]'):
        raise Exception('loopcfg: PTX_BRANCHY matches a load; it would refuse '
                        'every kernel')
    raw = [('i', 'bra $L_0'), ('i', '@%p0 brx.idx %r1, $T'), ('i', 'add.s32 %r1, %r2, %r3')]
    unk = ptx_unclassified_branches(raw)
    if [t for _, t in unk] != ['@%p0 brx.idx %r1, $T']:
        raise Exception(f'loopcfg: ptx_unclassified_branches reports {unk}, not the '
                        'one branch-family instruction PTX_BRA cannot parse')
    # the index helper must resolve a number and REFUSE a name -- not `int()` it,
    # which is the `invalid literal for int()` crash this directory records
    if ptx_pred_index('p7') != 7 or ptx_pred_index('rt_p0') is not None \
            or ptx_pred_index(None) is not None:
        raise Exception('loopcfg: ptx_pred_index does not separate %pN from a name')


_self_check()
