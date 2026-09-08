"""Does repairing the contraction unlock any kernel TODAY -- and which repair?

THIS FILE USED TO ANSWER "0 / 9" AND BOTH HALVES WERE WRONG.

  The 9 was a HARDCODED list, and it disagreed with `contract.py`'s own
  measurement in BOTH directions: it named six `gemm_f16_bias_relu_*` kernels
  where ptxas contracts nothing, and omitted ten kernels where it does.  Two
  lists of one thing drift; the set is derived here now.

  The 0 came from a hardcoded blocker table whose entry for every loop kernel
  was "needs loop invariants".  That was true when it was written and `loopval`
  has since provided them.  A tool that models another tool's answer instead of
  asking it goes stale silently -- so this asks.

AND THE REPAIR NAMED WAS THE EXPENSIVE ONE.  Forbidding the contraction with
`mul.rn.f32`/`add.rn.f32` costs 0 to +7.1% instructions and gives the kernel
TWO roundings where the hardware does one.  Saying `fma.rn.f32` instead -- the
fusion the machine already performs -- emits a BYTE-IDENTICAL instruction
stream, is more accurate, and validates.  The repair is to SAY what the machine
does, not to forbid it.
"""
import contextlib, glob, io, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or '.')
import contract, fpmode, gap, loopgap, loopval, ptxexec


def structural(k, o1=False):
    """(verdict, reason) from loopval, with its progress output suppressed.

    The VERDICT is returned separately from the reason because the pair below
    asserts an UNPROVED as well as a VALIDATED, and UNPROVED's message is the
    failing obligation ("store 0 value: sat") rather than the word.
    """
    d = 'o1' if o1 else 'corpus'
    p, s = f'{d}/{k}.ptx', f'{d}/{k}.sass'
    if not os.path.exists(s): return ('-', 'no build')
    if not loopgap.has_control_flow(p): return ('-', '(no loop)')
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            v, msg, _ = loopval.validate(p, s, 20, 'wide')
    except Exception as e:
        v, msg = 'REFUSED', str(e)
    return (v, v if v == 'VALIDATED' else loopgap.reason_key(msg))


# EVERY FLOAT-SEMANTIC OPCODE, not the fusion family.
#
# This used to be `(mul|add|sub|neg|fma)\.f32` -- "the family a change to the
# fusion path moves WITHIN".  That scope is right about fusions and it is not
# the scope of the defect below: an emitter change can hand the validator an
# opcode it refuses in ANY family, and the narrow regex counted 5 of the 29
# float-valued opcodes the emitter actually writes.  Four of the uncounted ones
# had reach at or above the 7 of `neg.f32`, the opcode the gate exists for.
#
# `ld.global.v4.f32` and friends are deliberately OUT: they move a bit pattern
# and are unmodelled for a VECTOR-WIDTH reason, not a floating-point one.  The
# boundary is "does the result depend on interpreting the bits as a float".
FLOAT_TY = ('f16', 'f32', 'f64', 'f16x2', 'bf16',
            'e4m3', 'e5m2', 'e4m3x2', 'e5m2x2')
SEM_MNEM = frozenset(('mul', 'add', 'sub', 'neg', 'fma', 'mad', 'div', 'rcp',
                      'sqrt', 'rsqrt', 'abs', 'max', 'min', 'ex2', 'lg2', 'sin',
                      'cos', 'tanh', 'setp', 'selp', 'cvt', 'testp', 'copysign'))
OPCODE = re.compile(r'^\s*([a-z][a-z0-9._]*)\s')


def is_float_semantic(op):
    parts = op.split('.')
    return parts[0] in SEM_MNEM and any(p in FLOAT_TY for p in parts[1:])


# Why each unmodelled family is unmodelled.  A gate whose exclusion list has no
# reasons is a TODO list; the point of writing them down is that the next
# opcode the emitter starts writing matches NONE of them and fails.
#
# The macro-op family is DERIVED from fpmode.MACRO_OPS rather than listed here:
# two lists of one thing drift, which is the shape of bug this whole directory
# is about.
def why_unmodelled(op):
    if op in fpmode.MACRO_OPS:
        return ('macro-op', 'approximate or multi-instruction; ptxas seeds it '
                'with a MUFU or expands it into a Newton sequence with its own '
                'control flow, and identifying the two is a semantic claim a '
                'device probe would have to settle -- see fpmode.py')
    parts = op.split('.')
    if parts[0] == 'cvt':
        return ('conversion', 'a rounding operation between two float formats '
                '(or a float and an integer); each rounding mode needs its own '
                'device referee, exactly as max.f32 did')
    if 'f64' in parts:
        return ('f64', 'a second float domain; fpmode has one 32-bit sort, so '
                'this needs a parallel set of symbols and its own referees')
    if any(p in ('e4m3', 'e5m2', 'e4m3x2', 'e5m2x2') for p in parts):
        return ('fp8', 'saturating narrow-float conversion; the saturation is '
                'part of the semantics and is not measured')
    return (None, None)


def float_ops_the_emitter_writes():
    """Every float-semantic opcode present in a committed artifact, with one
    real instruction line for each.  Read off the ARTIFACTS rather than listed:
    a list of what the emitter emits is a second copy of the emitter."""
    out = {}
    files = sorted(glob.glob(os.path.join('..', '..', 'tests', '*.ptx')))
    for f in files:
        for ln in open(f):
            m = OPCODE.match(ln)
            if m and is_float_semantic(m.group(1)):
                out.setdefault(m.group(1), ln.strip().rstrip(';'))
    return files, out


def classify(files, ops):
    """MODELLED / excluded-by-family / unclassifiable, as ONE code path.

    Shared with the positive control below on purpose: a control that
    re-implements the classification is a second measurement, and two copies
    agree while both are wrong."""
    modelled, excluded, bad = [], {}, []
    for op, line in sorted(ops.items()):
        st = ptxexec.Ptx(gap.fresh(files[0]))
        try:
            st.step(line)
            modelled.append(op)
            continue
        except Exception:
            # ANY refusal means not modelled.  The first version of this gate
            # exempted a refusal that was not an OPCODE_ERR, on the reading that
            # the sample line's operands were at fault -- and that classified
            # `setp.lt.f64` as MODELLED, because it refuses on `%fd1`, a 64-bit
            # float register the executor has no sort for.  An opcode that
            # cannot be executed is not modelled whatever the message says; the
            # exemption was a guess in the convenient direction, which is the
            # one thing this directory refuses to do.
            pass
        fam, _ = why_unmodelled(op)
        if fam is None: bad.append(op)
        else: excluded.setdefault(fam, []).append(op)
    return modelled, excluded, bad


def check_the_gate_would_notice():
    """Would this gate SEE a new float opcode?  An all-clear is what a broken
    classification reports too, so the census below is worth exactly what this
    control is worth.

    Two synthetic cases, through the SAME classify() the census uses:

      `abs.f32` -- a real PTX opcode the emitter does not write today.  Not
      modelled, not a macro-op, not a conversion, not f64, not fp8, so it must
      be REPORTED.  That is the whole claim of the widened gate.

      `setp.lt.f64` -- refuses on its OPERAND rather than on its opcode.  The
      first version of this gate exempted that and called it MODELLED.  It must
      land in a family instead.
    """
    files = sorted(glob.glob(os.path.join('..', '..', 'tests', '*.ptx')))
    if not files:
        print('FAIL: no artifacts to build a control from'); return 1
    probe = {'abs.f32': 'abs.f32 %f1, %f2',
             'setp.lt.f64': 'setp.lt.f64 %p1, %fd1, %fd2'}
    modelled, excluded, bad = classify(files, probe)
    rc = 0
    if 'abs.f32' not in bad:
        print(f'FAIL: the control opcode abs.f32 was not reported '
              f'(modelled={modelled}, excluded={excluded}) -- this gate cannot '
              f'see a new float opcode, so its all-clear means nothing')
        rc = 1
    if 'setp.lt.f64' in modelled:
        print('FAIL: setp.lt.f64 counted as MODELLED -- it refuses on its '
              'operand, and treating an operand-shaped refusal as "the opcode '
              'is fine" is the guess this directory refuses to make')
        rc = 1
    if not rc:
        print('  control: abs.f32 is reported, setp.lt.f64 is not called modelled')
    return rc


def check_the_emitter_cannot_grow_the_gap():
    """An emitter change can hand the validator an opcode it refuses, and
    nothing measured that.

    It happened.  Repairing the RoPE rotation to say `fma.rn.f32` -- byte-
    identical SASS, same PTX instruction count -- replaced a `sub.f32`, which
    `ptxexec` models, with a `neg.f32`, which it did not.  `neg.f32`'s reach
    across the corpus went 4 kernels to 7, and the commit that did it carried a
    13-row mutation table and nine checks, none of which read the validator.

    THE FIRST VERSION OF THIS GATE HAD THE SAME SHAPE OF HOLE, and it was mine.
    It counted the FUSION family -- `(mul|add|sub|neg|fma).f32`, five opcodes --
    and the emitter writes THIRTY.  `max.f32` sat outside it at reach 10,
    HIGHER than the `neg.f32` the gate was written for.

    So the rule is total now: every float-semantic opcode in a committed
    artifact is either MODELLED or in a named family with a written reason.  A
    thirty-first is in neither and fails -- which is what the control above
    measures rather than assumes.
    """
    files, ops = float_ops_the_emitter_writes()
    # FLOOR.  A scan that reads nothing reports no unmodelled opcodes, perfectly.
    if len(files) < 20 or len(ops) < 20:
        print(f'FAIL: scanned {len(files)} artifacts and found {len(ops)} '
              f'float-semantic opcodes -- there is nothing to check')
        return 1
    modelled, excluded, bad = classify(files, ops)
    print(f'  {len(ops)} float-semantic opcodes across {len(files)} committed artifacts')
    print(f'    modelled ({len(modelled)}): {", ".join(modelled)}')
    for fam in sorted(excluded):
        print(f'    {fam} ({len(excluded[fam])}): {", ".join(excluded[fam])}')
    if bad:
        print(f'FAIL: the emitter writes {", ".join(bad)} and it is neither '
              f'modelled nor in a named family. An emitter change that is free '
              f'in instructions and free in SASS bytes is not free in the third '
              f'currency: what the validator can read.')
        return 1
    return 0


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
    ks = sorted(contract.contraction_kernels())
    # ANTI-DRIFT.  The set must be what the artifacts say, not a list someone
    # keeps up to date: the list this file used to carry was wrong in BOTH
    # directions (six kernels named that contract nothing, ten omitted).  A
    # hardcoded set is invisible to every assertion below -- the pair at the
    # bottom does not depend on it -- so the agreement is checked here.
    measured = sorted(contract.contraction_kernels())
    if ks != measured:
        print(f'FAIL: fpgate is not measuring its own set -- it reports {len(ks)} '
              f'kernels where contract.py measures {len(measured)}')
        return 1
    # ...and the guard above compares one function against ITSELF, so it catches
    # a hardcoded LIST and is SILENT about a wrong MEASUREMENT.  Both sides move
    # together, which is the same silence a generated description has.
    #
    # THE SET IS NOW EMPTY, which makes that silence total: a measurement
    # computing nothing returns exactly what a corpus contracting nowhere
    # returns.  So the POSITIVE control can no longer be a shipped kernel and
    # is a PERTURBED one -- split a shipped `fma.rn.f32` back into the
    # `mul.f32` + `add.f32` it replaced and require the metric to flag it.
    # Same device as keeping `naive_gemm_f32_muladd` in the corpus.
    live, why = contract.measurement_is_live()
    if not live:
        print(f'FAIL: the contraction measurement is not live -- {why}')
        return 1
    print(f'  positive control: {why}')
    # The NEGATIVE control keeps the metric from over-reporting, and pins a
    # direction a previous reading got wrong: swiglu does NOT fuse.  FFMA is 0
    # in both builds and what moves under `.rn` is FSEL 4 -> 8 and IMAD
    # 202 -> 206, a scheduling difference.  The published claim that it fused
    # at HALF precision was a hypothesis asserted as fact; `HFMA2` appears in
    # neither build.  "The SASS moves" is not "ptxas fused".
    for k, want in (('gemm_f16_swiglu_512', False),):
        if not os.path.exists(f'corpus/{k}.ptx'): continue
        if (k in measured) != want:
            print(f'FAIL: {k} is {"absent from" if want else "in"} the contraction '
                  f'set. That is the metric answering by inference rather than by '
                  f'forbidding the fusion and re-assembling.')
            return 1
    print(f'{len(ks)} kernels where ptxas actually fuses a mul.f32 into an add.f32\n')
    # The doc quotes this number, and it has been wrong twice -- once as a
    # hardcoded 9, once as a derived 16.  Check it here, where the measurement
    # is, rather than leaving a third copy to go stale on its own.
    doc = os.path.join('..', '..', 'docs', 'ptxas_translation_validation.md')
    if os.path.exists(doc):
        m = re.search(r'\*\*Contraction\*\* [^\n]*?\*\*(\d+) kernels\*\*', open(doc).read())
        if not m:
            print('FAIL: the doc no longer states a contraction count for this to check')
            return 1
        if int(m.group(1)) != len(ks):
            print(f'FAIL: the doc says {m.group(1)} kernels contract; measured {len(ks)}')
            return 1
    print(f'{"kernel":38s}{"gap":>4}   what still blocks it')
    unlocked = []
    for k in ks:
        r = gap.census(k)
        pg = [o for o in r['ptx'][1] if o != 'bra']   # loopval owns `bra`
        sg = r['sass'][1]
        n = len(pg) + len(sg)
        v, why = structural(k)
        if n: why = f'{n} unmodelled opcode(s); ' + why
        print(f'{k:38s}{n:4d}   {why}')
        if n == 0 and v == 'VALIDATED': unlocked.append(k)

    # The one kernel a repaired build exists for -- and the PAIR is the result,
    # so both halves are asserted rather than just the green one.  The emitter
    # says `fma.rn.f32` now, so the SHIPPED kernel is the validated one and the
    # refutation lives on in `_muladd`, the form it used to emit.  Asserting
    # only the green half would be satisfied by a validator that never refutes.
    print()
    bad = 0
    for v, want in (('naive_gemm_f32', 'VALIDATED'), ('naive_gemm_f32_muladd', 'UNPROVED')):
        got, why = structural(v, o1=True)
        mark = 'ok' if got == want else f'CHANGED (wanted {want})'
        print(f'  o1/{v:24s} {got:10s} {mark:28s} {why if got != "VALIDATED" else ""}')
        if got != want: bad += 1
        if got == 'VALIDATED' and not v.endswith('_muladd'): unlocked.append(v)
    # A printed CHANGED is not a gate.  This pair is a STANDING RESULT in both
    # directions -- the shipped kernel must validate, and the form it replaced
    # must still be refuted, because a corpus containing nothing the validator
    # refutes cannot be told apart from a validator that always says VALIDATED.
    if bad:
        print(f'\nFAIL: {bad} of the two standing verdicts moved.')
        return 1

    print()
    # The control FIRST: the census below is worth exactly what it is worth.
    if check_the_gate_would_notice():
        return 1
    if check_the_emitter_cannot_grow_the_gap():
        return 1

    print(f'\nkernels the fma.rn repair unlocks today: {len(unlocked)}'
          f'{" -- " + ", ".join(unlocked) if unlocked else ""}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
