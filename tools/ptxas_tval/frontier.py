"""Which kernels are ONE feature from validating -- the SUFFICIENCY census.

`gap.py --rank` gives two columns and neither answers the question a roadmap
asks.  COST is how many opcodes a given kernel is short.  REACH is how many
kernels an opcode BLOCKS.  Reach is the one that reads like a ranking and it
has been wrong twice here, in the same direction: `max.f32` blocks ten kernels
and unblocks none, and the tensor-core staging blocks twenty-three and unblocks
none.  Both crosses were done by hand, in prose, and neither is re-derivable.

A kernel has more than one KIND of blocker, gated by different layers, so a
sufficiency census has to cross three measurements.  All three are imported;
none is restated.  A second implementation of an aggregation agrees with the
thing it is checking while both are wrong -- the recorded failure mode of an
agreement gate whose two sides move together.

  opcode      `gap.census`.  `bra` is DISCOUNTED from the PTX side.  It is not
              an unmodelled opcode: it is the straight-line executor being
              handed a loop, and `loopval` is the layer that handles one.
              Counting it inflates every loop kernel by one AND, worse, dresses
              a structural blocker up as a feature gap.  The SASS side keeps
              its branch opcodes, because `sassexec` does model `BRA`.

  structure   `loopgap`, for any kernel with control flow.  A kernel whose
              every opcode is modelled can still be refused because its loop
              shape is outside what the validator handles, and that is a
              separate blocker that no opcode work removes.  `loopgap` asks the
              SUITE -- `loopval` for one loop, `nestval` for one nest.  It used
              to ask `loopval` alone, which is a FIRST-REFUSAL reading of this
              column and is what listed `y_cpu_matmul` at -O1 as one blocker
              away from a structure `nestval` VALIDATES.

  wall        `wall.verdict`, the solver wall as a PROXY: the worst barrier
              region's symbolic integer multiply count against thresholds
              DERIVED from named region-level ground truth.  Without it the
              frontier called nine kernels clear at -O3 and four of those have
              never validated -- so "clear" was an upper bound and any ranking
              on top of it, a joint-sufficiency one above all, inherited that.
              A blocker is added only where the proxy says PAST; UNDECIDED is
              named as a lower bound, like the unroll layer's refusals.

  setup       the census could not build an initial state, so it executed
              nothing and reports an EMPTY opcode gap.  That is not a gap of
              zero and must never be ranked as one; it is its own blocker.

WHAT IT PRINTS.  The FRONTIER (kernels by blocker count, nearest first) and
SUFFICIENCY (for each blocker, the number of kernels it is the SOLE blocker
of, beside the number it blocks at all).  A blocker whose sole-count is zero
is necessary for something and sufficient for nothing, which is the sentence
this file exists to make cheap.

THE OPTIMISATION LEVEL IS PART OF THE QUESTION, and `--o1` is why.  The doc
records "a structural refusal moves with the optimisation level and an opcode
gap does not".  The first clause is right; the second is FALSE and this file is
what measures it -- `y_cpu_matmul`'s SASS gap is `PLOP3.LUT, UIADD3` in the
committed corpus and EMPTY at `-O1`.  So a sufficiency answer taken at one level
is an answer for that level, and asking at both is what turns "sufficient for
nothing" into a ranking.  `--o1` re-assembles every corpus `.ptx` at `-O1`, each
AT ITS OWN DECLARED TARGET (assembling at the build machine's arch is the bug
`ptx_portability.rs` exists to prevent, and here it would silently change which
SASS is under test), and runs the same census over that.

THE STRUCTURAL CENSUS RUNS IN ITS OWN PROCESS, and that is not tidiness -- see
`structural`.  Running it inline after the opcode census turned a standing
VALIDATED result into a REFUTATION.

THE CACHE REFUSES TO SERVE STALE DATA.  `gap.census` over the corpus is minutes,
so the measurement is cached -- and a cache that silently answers for an older
tree is this repository's own `.ysu_hw_profile` trap.  The key is a digest of
every corpus artifact, every executor source AND THIS FILE, so a changed
emitter, a changed model or a changed taxonomy invalidates it by name rather
than by remembering to pass a flag.  The first version left this file out and
served a contaminated answer straight back after the repair above.

THE CONTROLS, and all four perturb the INPUT.  An all-clear is what a census
that measured nothing also reports, so: the floor counts kernels examined; the
census handed two kernels must report two; a clear kernel must stop being clear
when its `.ptx` gains one unmodelled opcode; the cache key must name every
module that decides the answer; and the isolation control asserts BOTH that the
in-process census is still contaminated (or it is vacuous and says so) and that
the isolated one is not.  A control applied to the ANSWER cannot see the
measurement being subverted to read something else -- that hole was found by
mutation in `docgate.py`, one file over, on the day it was written.
`--selftest` runs the four in about ten seconds, without the census.
"""
import collections, glob, hashlib, json, os, re, sys

import gap
import loopgap
import unroll
import wall

# The unroll layer's verdict per kernel, recorded by `measure` IN THE DIRECTORY
# IT READ.  See `unroll_unknown` for why this may not be re-derived later.
_UNROLL = {}
_WALL = {}

CACHE = '.frontier_cache{}.json'
# `bra` is loopval's layer, not an unmodelled opcode -- see the docstring.
PTX_DISCOUNT = {'bra', 'bra.uni'}
# The executor sources whose behaviour the census reports.  A change to any of
# them changes the answer, so a cache taken before it is not an answer.
# EVERY module whose content decides the answer, DERIVED from this file's own
# import closure rather than listed.  A hand-maintained list is the defect this
# directory keeps finding, and it was here twice over: the first version left out
# THIS file, where the blocker taxonomy lives, and the second left out
# `loopcfg.py`, which decides every structural blocker.  Deriving it then found
# two more nobody had noticed -- `conc.py` and `mac64.py`, reached through
# `loopval`'s own comma-separated import.
#
# A cache that silently answers for an older tree is this repository's own
# `.ysu_hw_profile` trap, so the key must name the modules by construction.
MODELS = None   # set below, once _local_closure is defined


def _local_closure(root):
    """Every local .py reachable from `root` by import, `root` included.

    Textual rather than by importing: this runs inside a control, and importing
    a module to find out whether it should be in a digest is a side effect in
    the middle of a check."""
    here = os.path.dirname(os.path.abspath(__file__))
    seen, todo = set(), [root]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        path = os.path.join(here, m)
        if not os.path.exists(path):
            continue
        seen.add(m)
        for line in open(path):
            line = line.strip()
            # `import a, b, c` names THREE modules.  Capturing only the first
            # under-reports the closure, which makes this control weaker than
            # it looks -- it missed `sassexec.py`, reached through `loopval`'s
            # own comma-separated import.
            mm = re.match(r'^import\s+([\w.,\s]+)', line) or \
                 re.match(r'^from\s+([\w.]+)\s+import', line)
            if not mm:
                continue
            for part in mm.group(1).split(','):
                cand = part.strip().split('.')[0].split(' ')[0] + '.py'
                if os.path.exists(os.path.join(here, cand)):
                    todo.append(cand)
    return seen


MODELS = sorted(_local_closure('frontier.py'))


def digest():
    h = hashlib.sha256()
    for p in sorted(glob.glob('corpus/*.ptx') + glob.glob('corpus/*.sass')) + sorted(MODELS):
        h.update(p.encode())
        h.update(open(p, 'rb').read())
    return h.hexdigest()


def measure(only=None):
    """The three censuses, crossed.  `only` restricts the corpus.

    `only` exists so the measurement's own INPUT can be perturbed by a control
    that goes through this same call.  A control that perturbs the RESULT shows
    the comparison is live and says nothing about what the measurement read."""
    ks = sorted(os.path.basename(x)[:-4] for x in glob.glob('corpus/*.ptx'))
    if only is not None:
        ks = [k for k in ks if k in only]
    rows = {}
    for n, k in enumerate(ks, 1):
        # Progress on STDERR, because a census that takes minutes and prints
        # nothing cannot be told from one that stopped -- a trap this
        # repository has hit twice with detached sweeps.  stdout stays the
        # measurement, so a gate reading it is unaffected.
        print(f'[{n}/{len(ks)}] {k}', file=sys.stderr, flush=True)
        r = gap.census(k)
        blockers, setup = [], []
        for side in ('ptx', 'sass'):
            first, opgap, _ = r[side]
            if str(first).startswith('setup:'):
                setup.append(f'{side} setup: {str(first)[7:47].strip()}')
                continue
            for o in opgap:
                if side == 'ptx' and o in PTX_DISCOUNT:
                    continue
                blockers.append(f'{side}:{o}')
        blockers += setup
        rows[k] = sorted(set(blockers))
    for k, reason in structural(ks).items():
        rows[k] = sorted(set(rows[k] + ['loop:' + reason]))
    # THE FOURTH LAYER.  `loopval`'s relation holds at the loop header, so it
    # needs the two loops in lockstep; a kernel whose SASS loop composes the
    # PTX body k times is blocked by that whatever its opcodes say.  For eight
    # increments this file crossed three layers and reported the result as a
    # ranking, which made every distance a LOWER BOUND.
    #
    # A blocker is added only where the proxy DECIDES.  A refusal is the
    # instrument failing to answer, and recording it as a blocker would be a
    # guess in the pessimistic direction -- so those kernels are reported by
    # `unroll_unknown` instead, which is what makes the remaining lower bound
    # explicit rather than silent.
    _UNROLL.clear()
    for k in ks:
        v, det = unroll.factor(k)
        _UNROLL[k] = (v, det)
        if v == 'UNROLLED':
            rows[k] = sorted(set(rows[k] + ['unroll:' + unroll.UNROLL_KEY]))
    # THE FIFTH LAYER, crossed the same way and for the same reason: a blocker
    # only where the proxy DECIDES.  PTX-side, so it answers identically at -O1.
    _WALL.clear()
    for k in ks:
        v, det = wall.verdict(k)
        _WALL[k] = (v, det)
        if v == 'PAST':
            rows[k] = sorted(set(rows[k] + ['wall:' + wall.WALL_KEY]))
    return rows


def wall_unknown(rows=None):
    """Kernels the wall proxy cannot decide -- READ FROM `_WALL`, for the reason
    `unroll_unknown` records: re-measuring here would read whatever corpus is
    underfoot, which after the -O1 census is not the one measured."""
    ks = sorted(rows) if rows is not None else sorted(_WALL)
    missing = [k for k in ks if k not in _WALL]
    if missing:
        raise Exception(f'the wall layer was not measured for {missing[:3]}')
    return {k: _WALL[k][1] for k in ks if _WALL[k][0] in ('REFUSED', 'UNDECIDED')}


def unroll_unknown(rows=None):
    """Kernels whose unroll status the proxy cannot decide.

    For these -- and only these -- the distance `report` prints is still a lower
    bound.  Naming them is the difference between a ranking that states its own
    limit and one that does not.

    READ FROM `_UNROLL`, NEVER RE-MEASURED, and that is not a cache.  The -O1
    census runs `measure` inside a TEMPORARY directory of -O1 artifacts and
    chdirs back before `report` and `controls` run, so re-measuring here reads
    the -O3 corpus and answers about the wrong artifact.  Caught by the
    consistency control below, which compared an -O1 clear set against an -O3
    unroll verdict and said so -- this increment's own subject, committed in the
    control written to catch it.  A kernel measured by nobody RAISES rather than
    being silently answered from whatever corpus happens to be underfoot."""
    ks = sorted(rows) if rows is not None else sorted(_UNROLL)
    missing = [k for k in ks if k not in _UNROLL]
    if missing:
        raise Exception(f'the unroll layer was not measured for {missing[:3]} -- '
                        '`measure` records it in the directory it read, so asking '
                        'here without a census would answer about another tree')
    return {k: _UNROLL[k][1] for k in ks if _UNROLL[k][0] in ('REFUSED', 'UNDECIDED')}


def structural(ks):
    """`loopgap`'s refusal per control-flow kernel, IN A SEPARATE PROCESS.

    THE PROCESS BOUNDARY IS LOAD-BEARING, and it is here because running the
    two censuses in one interpreter CHANGED A VERDICT.  The first version of
    this file called `loopgap.census([k])` inline, straight after `gap.census`,
    and over a full corpus that reported `exact_pv` at -O1 as
    `store 0 value: sat` -- a REFUTATION of a kernel that is a standing
    VALIDATED result and that `loopgap` alone, on byte-identical inputs,
    validates.  Not a budget artifact: it validates at budget 20 and 60 when
    `loopgap` runs on its own.

    The mechanism is the one `sassexec.run_insns` already warns about in its
    seeding comment: the multiply primitive canonicalises its operands by z3
    NODE ID at construction, and node ids come from a global counter, so
    unrelated work done earlier in the same interpreter can leave two terms
    with their operands in opposite orders and congruence closure will not
    relate them.  One preceding kernel is not enough to move it; a corpus of
    them is.

    So the structural column is measured the way `loopgap.py` measures it --
    one process, that census and nothing else -- which also makes this column
    equal to the doc's census by construction rather than by coincidence."""
    import json, subprocess
    want = [k for k in ks if loopgap.has_control_flow(f'corpus/{k}.ptx')]
    if not want:
        return {}
    # Progress on the child's STDERR for the same reason the opcode loop prints
    # it: this phase is minutes and silent, and a silent phase cannot be told
    # from a stopped one.
    src = ('import json,sys; import loopgap; out={}\n'
           'for n,k in enumerate(sys.argv[1:],1):\n'
           '    print(f"[loop {n}/{len(sys.argv)-1}] {k}", file=sys.stderr, flush=True)\n'
           '    r = loopgap.census([k])[0]\n'
           '    if r[1] != "VALIDATED": out[r[0]] = loopgap.reason_key(r[2], r[0])\n'
           'print(json.dumps(out))')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get('PYTHONPATH', '')
    r = subprocess.run([sys.executable, '-c', src] + want,
                       capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit('FAIL: the structural census subprocess failed:\n' + r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def build_at(ptx, level, outdir):
    """Assemble one `.ptx` at `-O<level>` AT ITS OWN DECLARED TARGET.

    The target comes from the artifact's own `.target` line rather than from
    this machine's card: assembling at the build machine's arch is the bug
    `ptx_portability.rs` exists to prevent, and here it would quietly change
    which SASS is under test.  Returns the disassembly, or None."""
    import subprocess
    m = re.search(r'\.target\s+(sm_\d+)', open(ptx).read())
    if not m:
        return None
    cub = os.path.join(outdir, os.path.basename(ptx)[:-4] + f'.O{level}.cubin')
    if subprocess.run(['ptxas', f'-O{level}', '-arch=' + m.group(1), '-o', cub,
                       os.path.abspath(ptx)], capture_output=True).returncode:
        return None
    r = subprocess.run(['nvdisasm', '-c', cub], capture_output=True, text=True)
    return None if r.returncode else r.stdout


def gap_at(kernel, level=None):
    """The OPCODE gap of one kernel, at the committed level or re-assembled.

    `level=None` reads the committed corpus; an integer re-assembles that
    kernel's own `.ptx` at that `-O` and censuses the result.  `bra` is
    discounted from the PTX side for the reason in the module docstring.

    Split out so `docgate.py` can gate the doc's `-O` figures against the same
    call this file measures with, rather than growing a second implementation
    of "build it and ask the census" that would agree with the doc while both
    were wrong."""
    import shutil, tempfile
    if level is None:
        r = gap.census(kernel)
    else:
        here = os.getcwd()
        d = tempfile.mkdtemp(prefix='frontier_gapat_')
        try:
            os.makedirs(os.path.join(d, 'corpus'))
            sass = build_at(f'corpus/{kernel}.ptx', level, d)
            if sass is None:
                raise SystemExit(f'FAIL: could not assemble {kernel} at -O{level}')
            os.symlink(os.path.abspath(f'corpus/{kernel}.ptx'),
                       os.path.join(d, 'corpus', kernel + '.ptx'))
            open(os.path.join(d, 'corpus', kernel + '.sass'), 'w').write(sass)
            os.chdir(d)
            r = gap.census(kernel)
        finally:
            os.chdir(here)
            shutil.rmtree(d, ignore_errors=True)
    return {'ptx': [o for o in r['ptx'][1] if o not in PTX_DISCOUNT],
            'sass': list(r['sass'][1])}


def o1_corpus():
    """Re-assemble every corpus `.ptx` at -O1, each at ITS OWN declared target.

    Returns a directory holding a `corpus/` the census can be pointed at by
    chdir.  The target comes from the artifact's own `.target` line rather than
    from this machine's card: assembling at the build machine's arch is the bug
    `ptx_portability.rs` exists to prevent, and here it would quietly change
    which SASS is under test."""
    import tempfile
    d = tempfile.mkdtemp(prefix='frontier_o1_')
    os.makedirs(os.path.join(d, 'corpus'))
    built = 0
    for p in sorted(glob.glob('corpus/*.ptx')):
        k = os.path.basename(p)[:-4]
        sass = build_at(p, 1, d)
        if sass is None:
            continue
        os.symlink(os.path.abspath(p), os.path.join(d, 'corpus', k + '.ptx'))
        open(os.path.join(d, 'corpus', k + '.sass'), 'w').write(sass)
        built += 1
    if not built:
        raise SystemExit('FAIL: could not assemble any kernel at -O1')
    return d, built


def load(only=None, o1=False):
    """Cached, and the cache is refused rather than trusted when the tree moved."""
    if only is not None:
        return measure(only)                      # a perturbed input is never cached
    cache = CACHE.format('_o1' if o1 else '')
    d = digest() + ('|o1' if o1 else '')
    if os.path.exists(cache):
        c = json.load(open(cache))
        if c.get('digest') == d:
            return c['rows']
        print(f'note: cache is for a different tree ({c.get("digest","?")[:12]} != '
              f'{d[:12]}); re-measuring', file=sys.stderr)
    if o1:
        here = os.getcwd()
        tmp, built = o1_corpus()
        print(f'note: assembled {built} kernels at -O1', file=sys.stderr)
        try:
            os.chdir(tmp); rows = measure()
        finally:
            os.chdir(here)
    else:
        rows = measure()
    json.dump({'digest': d, 'rows': rows}, open(cache, 'w'), indent=1)
    return rows


def report(rows):
    blocks = collections.Counter()
    sole = collections.Counter()
    for k, bs in rows.items():
        for b in bs:
            blocks[b] += 1
        if len(bs) == 1:
            sole[bs[0]] += 1
    clear = sorted(k for k, bs in rows.items() if not bs)

    print(f'=== FRONTIER: {len(rows)} kernels, {len(clear)} with no blocker ===')
    for k in clear:
        print(f'  0  {k}')
    for k, bs in sorted(rows.items(), key=lambda kv: (len(kv[1]), kv[0])):
        if not bs:
            continue
        print(f'{len(bs):3d}  {k}')
        for b in bs:
            print(f'       {b[:100]}')

    print('\n=== SUFFICIENCY: sole blocker / blocks at all ===')
    print(f'{"sole":>5} {"blocks":>7}  blocker')
    for b, n in blocks.most_common():
        print(f'{sole[b]:5d} {n:7d}  {b[:88]}')
    if not sum(sole.values()):
        print('\nNo single feature validates any kernel: the frontier is empty at distance 1.')
    unk = unroll_unknown(rows)
    if unk:
        print(f'\n=== STILL A LOWER BOUND for {len(unk)} of {len(rows)} kernels ===')
        print('The unroll proxy cannot decide these, so a fourth blocker may stand')
        print('behind the distance printed above.  It is named, not assumed.')
        for k in sorted(unk):
            print(f'  {k:46s} {unk[k][:70]}')
    wunk = wall_unknown(rows)
    if wunk:
        print(f'\n=== STILL A LOWER BOUND for {len(wunk)} of {len(rows)} kernels (wall) ===')
        print('The wall proxy cannot decide these -- their worst region sits between the')
        print('largest measured PROVED and the smallest measured UNKNOWN, or its work is')
        print('not the symbolic integer multiply the ground truth measures.')
        for k in sorted(wunk):
            print(f'  {k:46s} {wunk[k][:70]}')
    return blocks, sole, clear


def perturbed_corpus(kernel, extra):
    """Run the WHOLE measurement over a corpus in which one kernel gained `extra`.

    Perturbing the INPUT, through the same `measure` call.  A control applied to
    the answer shows the comparison is live and cannot see the measurement being
    subverted to read something else -- found by mutation in `docgate.py`.

    It is done by chdir into a symlink farm rather than by adding a directory
    parameter, and that is deliberate: `gap.census`, `loopgap.has_control_flow`
    and `loopgap.census` each open `corpus/...` themselves, so the cwd perturbs
    every consumer identically without changing three signatures."""
    import shutil, tempfile
    here = os.getcwd()
    tmp = tempfile.mkdtemp(prefix='frontier_ctl_')
    try:
        os.makedirs(os.path.join(tmp, 'corpus'))
        for f in glob.glob('corpus/*'):
            os.symlink(os.path.abspath(f), os.path.join(tmp, 'corpus', os.path.basename(f)))
        tgt = os.path.join(tmp, 'corpus', kernel + '.ptx')
        os.unlink(tgt)
        # Insert before the body's closing brace rather than before a `ret`.
        # Not every kernel here ends in one -- `bn254_fr_mul_fast` finishes on a
        # predicated store -- and the first version of this control crashed on
        # exactly that, which is a control that reports nothing.
        lines = open(f'corpus/{kernel}.ptx').read().splitlines()
        i = max(n for n, l in enumerate(lines) if l.strip() == '}')
        open(tgt, 'w').write('\n'.join(lines[:i] + ['\t' + extra] + lines[i:]) + '\n')
        os.chdir(tmp)
        return measure(only=[kernel])
    finally:
        os.chdir(here)
        shutil.rmtree(tmp, ignore_errors=True)


DOC = '../../docs/ptxas_translation_validation.md'


def doc_figures(o1, text=None):
    """The figures the doc publishes about THIS census, for `check_doc`.

    Whitespace-tolerant: both files wrap, and a sentence-shaped pattern with
    hard spaces in it matches only until somebody reflows a paragraph -- and
    then the gate reports the claim as MISSING rather than as wrong.

    `text` lets a control hand in a PERTURBED doc through this same parse."""
    d = open(DOC).read() if text is None else text
    one = _one_away(d, o1)
    if one is None:
        return None
    if o1:
        # THE COUNTS TOO, not just the sole-blocker set.  Asserting only the set
        # left `112 distinct blockers and 10 clear` published and gated by
        # NOTHING -- a gate that asserts the ROUTE and never the NUMBER, which
        # is the certificate-count defect one increment later, in the gate
        # written to stop published figures going stale.
        # EITHER form, and one of them is REQUIRED.  The sole-blocker set at
        # -O1 is empty now, and a parse that can only express "exactly one" has
        # no way to say so -- it would report the doc as MISSING the claim,
        # which is a gate failing rather than a gate checking.  The empty case
        # is a positive sentence in the doc for the same reason: a gate must not
        # accept the absence of a claim as agreement with it.
        n = re.search(r'At\s+`-O1`\s+the\s+corpus\s+is\s+(\d+)\s+kernels,\s+\*\*(\d+)\*\*\s+'
                      r'distinct\s+blockers\s+and\s+\*\*(\d+)\*\*\s+clear', d)
        if not n:
            return None
        return {'one': one, 'kernels': int(n.group(1)),
                'distinct': int(n.group(2)), 'clear': int(n.group(3))}
    a = re.search(r'(\d+)\s+kernels,\s+\*\*(\d+)\*\*\s+distinct\s+blockers', d)
    b = re.search(r'\*\*(\d+)\s+kernels\s+are\s+clear,\s+the\s+median\s+is\s+(\d+)\s+blockers'
                  r'\s+and\s+(\d+)\s+of\s+(\d+)\s+are\s+(\d+)\s+or\s+more\*\*', d)
    if not a or not b:
        return None
    return {'one': one, 'kernels': int(a.group(1)), 'distinct': int(a.group(2)),
            'clear': int(b.group(1)), 'median': int(b.group(2)),
            'tail_n': int(b.group(3)), 'tail_of': int(b.group(4)), 'tail_at': int(b.group(5))}


def _one_away(d, o1):
    """The set of kernels the doc says are ONE blocker away at a level, or None
    if it states no such claim.

    A SET, at BOTH levels.  The parse used to hardwire the -O3 answer as "every
    sole-count is zero" and the -O1 one as "exactly one kernel, or none", and the
    wall layer made both false at once: it is the sole blocker of four kernels at
    each level.  A parse that can only express the previous answer reports the
    new one as a MISSING claim, which is a gate failing rather than checking.
    The empty set is a positive sentence for the reason the old `-O1` form was:
    a gate must not accept the absence of a claim as agreement with it."""
    lvl = '1' if o1 else '3'
    z = re.search(r'at\s+`-O' + lvl + r'`\s+\*\*no\s+kernel\s+is\s+one\s+blocker\s+away\*\*', d)
    m = re.search(r'at\s+`-O' + lvl + r'`\s+the\s+kernels\s+one\s+blocker\s+away\s+are\s+'
                  r'((?:`\w+`(?:,\s*(?:and\s+)?|\s+and\s+)?)+)', d)
    if z and m:
        return None                  # contradictory claims are not a claim
    if z:
        return set()
    if m:
        return set(re.findall(r'`(\w+)`', m.group(1)))
    return None


STAMP = 'frontier_stamp.json'


def summarise(rows):
    """Everything `check_doc` compares, and nothing more.

    It is what the census STAMPS (`write_stamp`), so `docgate` can check the doc
    against the last measurement in seconds without re-running minutes of census.
    The stamp is a measurement, not a certificate: it is written whether or not
    the doc agrees, and the comparison is made by `compare_doc` on both paths."""
    return {'kernels': len(rows),
            'counts': sorted(len(b) for b in rows.values()),
            'distinct': len({b for bs in rows.values() for b in bs}),
            'sole': sorted(k for k, b in rows.items() if len(b) == 1)}


def write_stamp(rows, o1):
    """Record which TREE this measurement is of.

    WHY IT EXISTS.  The published figures were stale for two commits: a fixture
    rewrite moved them, nothing re-ran the census, and every gate run was
    `--selftest`, which never reaches `check_doc`.  A figure checked only by a
    minutes-long command is checked only when somebody pays for it.  With the
    digest stamped, `docgate` -- seconds, and run by every increment -- fails the
    moment the tree the figures were measured on stops being this tree."""
    s = json.load(open(STAMP)) if os.path.exists(STAMP) else {}
    s['O1' if o1 else 'O3'] = {'digest': digest() + ('|o1' if o1 else ''),
                               'summary': summarise(rows)}
    json.dump(s, open(STAMP, 'w'), indent=1, sort_keys=True)
    open(STAMP, 'a').write('\n')


def check_doc(rows, o1):
    """Assert, do not merely print.  A census figure that lands in a document and
    is checked by nothing decays the moment somebody edits an emitter -- this
    repository's own recorded lesson, and `docgate.py` exists because of it.
    `docgate` cannot pay for this one (the census is minutes), so the tool that
    already paid asserts its own published numbers, the way `regress.sh` does --
    and stamps them, so `docgate` can check them against the stamp instead."""
    return compare_doc(summarise(rows), o1)


def compare_doc(summ, o1, text=None):
    """The comparison, on a `summarise` result.  One implementation for the census
    and for `docgate`'s stamp check, so the two cannot disagree about the doc."""
    want = doc_figures(o1, text)
    if want is None:
        print('FAIL: the doc no longer states the figures this census publishes')
        return 1
    counts = summ['counts']
    sole = summ['sole']
    if o1:
        if set(sole) != want['one']:
            print(f'FAIL: doc says exactly {sorted(want["one"])} is one blocker away at -O1; '
                  f'measured {sorted(sole)}')
            return 1
        got1 = {'kernels': summ['kernels'],
                'distinct': summ['distinct'],
                'clear': sum(1 for c in counts if c == 0)}
        bad1 = {k: (want[k], got1[k]) for k in got1 if want[k] != got1[k]}
        if bad1:
            for k, (w, g) in sorted(bad1.items()):
                print(f'FAIL: doc says -O1 {k}={w}; measured {g}')
            return 1
        print(f'  doc: at -O1 exactly {sorted(sole)} is one blocker away, and the corpus is '
              f'{got1["kernels"]} kernels / {got1["distinct"]} distinct / {got1["clear"]} '
              'clear -- as published')
        return 0
    got = {'kernels': summ['kernels'], 'distinct': summ['distinct'],
           'clear': sum(1 for c in counts if c == 0),
           'median': counts[(len(counts) - 1) // 2],
           'tail_n': sum(1 for c in counts if c >= want['tail_at']),
           'tail_of': summ['kernels'], 'tail_at': want['tail_at']}
    bad = {k: (want[k], got[k]) for k in want if k != 'one' and want[k] != got[k]}
    if bad:
        for k, (w, g) in sorted(bad.items()):
            print(f'FAIL: doc says {k}={w}; measured {g}')
        return 1
    if set(sole) != want['one']:
        print(f'FAIL: the doc says {sorted(want["one"])} are one blocker away at -O3; '
              f'measured {sorted(sole)}')
        return 1
    print(f'  doc: {got["kernels"]} kernels, {got["distinct"]} distinct blockers, '
          f'{got["clear"]} clear, median {got["median"]}, {got["tail_n"]} at '
          f'{got["tail_at"]}+, one blocker away {sorted(sole)} -- as published')
    return 0


def controls(rows, clear):
    """The two positive controls, both perturbing the INPUT.

    Split out so a mutation harness can probe THEM without paying for the
    corpus census, which is minutes.  A control that perturbs the ANSWER
    instead cannot see the measurement being subverted to read something else;
    that hole was found by mutation in `docgate.py`, one file over."""
    # SNAPSHOT FIRST.  The controls below call `measure` again over a perturbed
    # corpus, and `measure` clears `_UNROLL` -- so reading it afterwards sees
    # the probe's two kernels rather than the census's.  Caught by the vacuity
    # leg of the layer-crossing control, which reported having nothing to
    # assert instead of passing.
    snap = dict(_UNROLL)
    wsnap = dict(_WALL)
    unr = sorted(k for k, (v, _d) in snap.items() if v == 'UNROLLED')
    two = sorted(rows)[:2]
    pert = measure(only=two)
    if sorted(pert) != two:
        print('FAIL: the census did not read the corpus it was given')
        return 1
    print(f'  control: handed {two}, the census reported those and no others')
    if not clear:
        print('FAIL: no kernel is clear, so the injection control has nothing to perturb')
        return 1
    k = clear[0]
    inj = perturbed_corpus(k, 'not.pred %p9, %p9;')
    got = inj.get(k, [])
    if not any('not.pred' in b for b in got):
        print(f'FAIL: {k} gained an unmodelled opcode and the census still reports {got};'
              ' it is not reading the artifact it was handed')
        return 1
    print(f'  control: {k} is clear, and reports {got} once its PTX gains one unmodelled opcode')
    # THE LAYER MUST NOT CONTRADICT THE CLEAR SET.  A kernel reported as having
    # no blocker is one the validator could plausibly reach; if the unroll layer
    # refused it, or called it unrolled, the frontier would be claiming a kernel
    # is zero features away while an uncounted layer blocked it -- which is
    # exactly the defect this layer was crossed in to remove.  Checkable
    # cheaply, and it holds at both levels: every clear kernel is either
    # loop-free or MATCHED.
    bad = {k: _UNROLL[k] for k in clear
           if _UNROLL.get(k, ('?',))[0] in ('UNROLLED', 'REFUSED', 'UNDECIDED')}
    if bad:
        print(f'FAIL: the frontier calls these kernels clear and the unroll layer '
              f'does not agree: {bad}')
        return 1
    print(f'  control: all {len(clear)} clear kernels are loop-free or 1:1 under the '
          f'unroll layer, so it contradicts none of them')
    # AND THE LAYER IS ACTUALLY CROSSED.  Deleting the four lines in `measure`
    # that add the blocker leaves every figure above plausible and every other
    # control green -- the census simply stops counting a layer, which is the
    # state this file was in for eight increments.  Asserted through `rows`,
    # which is what `measure` produced.
    # STATED AS A BICONDITIONAL, and that is what makes it non-vacuous at BOTH
    # levels.  "at least one kernel must be UNROLLED" is false of the -O1
    # corpus -- where all three of the layer's ground-truth kernels are
    # MATCHED, which is the correct state and not a defect -- so that rule
    # failed a census that was right.  A kernel carries the blocker IFF the
    # layer calls it UNROLLED: at -O3 that catches the crossing being removed,
    # at -O1 it catches a blocker being added to a kernel that is 1:1, and
    # neither direction is silent about the other.
    #
    # At -O1 there is no UNROLLED kernel, so the crossing itself is checked
    # there by nothing; it is checked by the -O3 census and by `--selftest`,
    # whose sub-corpus includes `exact_pv` for exactly that reason.
    wrong = []
    for k in sorted(snap):
        has = any(b.startswith('unroll:') for b in rows.get(k, []))
        want = snap[k][0] == 'UNROLLED'
        if has != want:
                wrong.append((k, snap[k][0], 'blocker' if has else 'no blocker'))
    if wrong:
        print(f'FAIL: the unroll verdict and the unroll blocker disagree for {wrong}; '
              f'the layer is measured and not crossed, or crossed where it decided '
              f'nothing')
        return 1
    print(f'  control: every one of {len(snap)} kernels carries an unroll blocker '
          f'exactly when the layer calls it UNROLLED ({len(unr)} do)')
    # THE WALL, the same biconditional.  Unlike unroll it holds non-vacuously at
    # BOTH levels: the proxy reads PTX, so its PAST set does not move with -O.
    wwrong = []
    for k in sorted(wsnap):
        has = any(b.startswith('wall:') for b in rows.get(k, []))
        if has != (wsnap[k][0] == 'PAST'):
            wwrong.append((k, wsnap[k][0], 'blocker' if has else 'no blocker'))
    if wwrong:
        print(f'FAIL: the wall verdict and the wall blocker disagree for {wwrong}')
        return 1
    past = sorted(k for k, (v, _d) in wsnap.items() if v == 'PAST')
    if not past:
        print('FAIL: no kernel is PAST the wall, so the crossing control asserts nothing')
        return 1
    # CALIBRATION through the census, not only in `wall --selftest`: no kernel the
    # standing results validate may carry a wall blocker.
    cal = sorted(k for k in wall.regress_validated() if any(b.startswith('wall:') for b in rows.get(k, [])))
    if cal:
        print(f'FAIL: regress.sh validates {cal} and the census gives them a wall blocker')
        return 1
    print(f'  control: every one of {len(wsnap)} kernels carries a wall blocker exactly when '
          f'the proxy says PAST ({len(past)} do), and none of them is a validated kernel')
    return cache_key_control() + isolation_control()


def cache_key_control():
    """Every module whose content decides the answer must be in the cache key.

    Source-level, because a cache serving a stale answer is only observable
    ACROSS runs and no single run can see it.  It is here because the first
    version of `MODELS` listed the executors and left out THIS file, where
    `PTX_DISCOUNT` and the blocker taxonomy live -- and the first run made after
    the process-boundary repair handed the contaminated answer straight back."""
    mine = os.path.basename(__file__)
    # `MODELS` IS the closure now, so asking whether it contains the closure is
    # vacuous.  What has to be checked is that the closure is COMPUTED, and the
    # control therefore perturbs its INPUT: a synthetic root importing a known
    # module, through the same call.  A closure that returned a constant, or
    # that stopped at the root, answers the real question perfectly.
    if mine not in MODELS:
        print(f'FAIL: {mine} decides the answer and is not in the cache key')
        return 1
    # the transitive case: `sassexec` is reached only through `loopval`, and
    # only by reading a comma-separated import -- the exact shape the first
    # version of this walk missed.
    for deep in ('loopcfg.py', 'sassexec.py', 'conc.py'):
        if deep not in MODELS:
            print(f'FAIL: {deep} is reachable by import and is not in the cache key')
            return 1
    import tempfile, shutil
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory(dir=here) as d:
        probe = os.path.join(os.path.basename(d), '_probe_root.py')
        with open(os.path.join(here, probe), 'w') as f:
            f.write('import gap, loopcfg\n')
        # the walk resolves names against `here`, so run it on a copy placed there
        tmpname = '_frontier_probe_root.py'
        shutil.copy(os.path.join(here, probe), os.path.join(here, tmpname))
        try:
            got = _local_closure(tmpname)
        finally:
            os.remove(os.path.join(here, tmpname))
    if not {'gap.py', 'loopcfg.py'} <= got:
        print(f'FAIL: a root importing gap and loopcfg produced the closure {sorted(got)}; '
              'the walk is not reading the file it was given')
        return 1
    print(f'  control: the cache key is the import closure, {len(MODELS)} sources '
          f'including {mine} itself; a synthetic root resolves to {sorted(got)}')
    return 0


def isolation_control():
    """The process boundary in `structural`, checked on the defect it exists for.

    BOTH DIRECTIONS, and the first one is the point: the hazard has to still be
    LIVE or the second says nothing.  `exact_pv` at -O1 is a standing VALIDATED
    result; after ~1.5s of unrelated census work the INLINE census stops
    validating it, and the isolated one does not.

    IT RUNS IN A CHILD, and that is not decoration.  The first version ran in
    this interpreter and PASSED under `--selftest` while going VACUOUS after a
    full census -- the contamination is an ordering effect, not a monotone one,
    so a control that inherits whatever the process has already done is not a
    controlled experiment.  It said so rather than passing quietly, which is
    what the "or it is vacuous" leg is for; the repair is to give it a fresh
    interpreter every time.

    AND IT SWEEPS THE PREAMBLE, because a fixed one went vacuous a second time.
    The first version used exactly 60 rounds, and a change that only altered what
    z3 terms get built elsewhere (the memory model's self-check) moved the hazard:
    exact_pv at -O1, inline verdict per preamble size, one fresh child each --

        preamble      0   5   12   30   60   120   240
        at e509590    V   U   U    V    U    V     U
        after it      V   U   U    V    V    U     U

    Non-monotone at BOTH commits, and 60 and 120 swapped.  The contamination is
    z3 node-id order, so any single count is a fact about today's construction
    history, not about the hazard.  The control tries PREAMBLES in order and uses
    the first that contaminates; only if none does is it vacuous."""
    import subprocess
    src = r"""
import os, sys, glob, tempfile, shutil, contextlib, io
import frontier, gap, loopgap
N = int(sys.argv[1])
d = tempfile.mkdtemp(prefix='frontier_iso_'); os.makedirs(d + '/corpus')
s = frontier.build_at('corpus/exact_pv.ptx', 1, d)
if s is None:
    print('BUILD-FAILED'); raise SystemExit(0)
os.symlink(os.path.abspath('corpus/exact_pv.ptx'), d + '/corpus/exact_pv.ptx')
open(d + '/corpus/exact_pv.sass', 'w').write(s)
for k in ('y_cpu_matmul', 'bn254_permute'):
    for e in ('.ptx', '.sass'):
        shutil.copy('corpus/' + k + e, d + '/corpus')
os.chdir(d)
for _ in range(N):
    gap.census('y_cpu_matmul'); gap.census('bn254_permute')
with contextlib.redirect_stdout(io.StringIO()):
    inline = loopgap.census(['exact_pv'])[0][1]
    isolated = frontier.structural(['exact_pv']) if inline != 'VALIDATED' else '-'
print('VERDICT', inline, '|', isolated)
"""
    env = dict(os.environ)
    env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get('PYTHONPATH', '')
    tried = []
    for n in PREAMBLES:
        r = subprocess.run([sys.executable, '-c', src, str(n)], capture_output=True, text=True, env=env)
        line = [l for l in r.stdout.splitlines() if l.startswith(('VERDICT', 'BUILD-FAILED'))]
        if not line:
            print('FAIL: the isolation control produced no verdict:\n' + r.stderr[-1500:])
            return 1
        if line[-1] == 'BUILD-FAILED':
            print('FAIL: could not assemble exact_pv at -O1 for the isolation control')
            return 1
        inline, isolated = (x.strip() for x in line[-1][len('VERDICT'):].split('|'))
        tried.append(f'{n}:{inline}')
        if inline == 'VALIDATED':
            continue
        if isolated not in ('{}', ''):
            print(f'FAIL: the isolated census refuses exact_pv at -O1 ({isolated}) after a '
                  f'{n}-round preamble; the process boundary is not doing its job')
            return 1
        print(f'  control: after a {n}-round preamble the in-process census says {inline} for '
              f'exact_pv at -O1 and the isolated one still validates it (tried {", ".join(tried)})')
        return 0
    print(f'FAIL: the in-process census still validates exact_pv after every preamble tried '
          f'({", ".join(tried)}), so this control is vacuous -- re-read why `structural` runs '
          'in its own process')
    return 1


# Tried in order; the first that contaminates is used.  Not one count, because the
# contaminating counts move whenever term construction elsewhere changes.
PREAMBLES = (5, 12, 60, 120, 240)



if __name__ == '__main__':
    if '--selftest' in sys.argv[1:]:
        # The CONTROLS alone, over a two-kernel corpus, for a mutation harness.
        # It probes the controls and not the census, and says so on stdout so a
        # reader cannot mistake its output for a frontier.
        ks = sorted(os.path.basename(x)[:-4] for x in glob.glob('corpus/*.ptx'))
        # `exact_pv` is in this sub-corpus DELIBERATELY: it is the kernel the
        # unroll layer blocks, and without it the layer-crossing control below
        # is vacuous on exactly the path a mutation harness runs.
        # `bn254_permute` is here because the first two are both blocked now --
        # `bn254_fr_mul_fast` by the wall -- and the injection control needs a
        # CLEAR kernel; `bn254_fr_mul_fast` is what makes the wall crossing
        # control non-vacuous on this path.
        sub = measure(only=ks[:2] + ['exact_pv', 'bn254_permute'])
        print('# controls only -- this is NOT the corpus census')
        sys.exit(controls(sub, sorted(k for k, b in sub.items() if not b)))
    o1 = '--o1' in sys.argv[1:]
    rows = load(o1=o1)
    print('# corpus as committed' if not o1 else '# every kernel re-assembled at -O1')
    # FLOOR.  A census that examined nothing ranks nothing, perfectly.
    if not rows:
        print('FAIL: examined no kernels -- there is nothing to rank'); sys.exit(1)
    blocks, sole, clear = report(rows)
    print('\n--- the doc figures this census publishes ---')
    bad = check_doc(rows, o1)
    write_stamp(rows, o1)
    print(f'  stamp: {STAMP} [{"O1" if o1 else "O3"}] records this measurement at tree '
          f'{digest()[:12]}; docgate compares the doc against it')
    print('\n--- positive controls ---')
    sys.exit(bad + controls(rows, clear))
