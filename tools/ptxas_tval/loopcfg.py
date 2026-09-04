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

# ---------------- PTX ----------------
PTX_LABEL = re.compile(r'^\$?([A-Za-z_][\w$]*)\s*:$')
PTX_BRA   = re.compile(r'^(?:@(!?)%p(\d+)\s+)?bra(?:\.uni)?\s+\$?([\w$]+)$')

def ptx_regions(path):
    raw = []
    started = False
    for line in open(path):
        s = line.strip()
        if s.startswith('//') or not s: continue
        if s == '{': started = True; continue
        if s == '}': break
        if not started: continue
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
    ins = lambda a, b: [t for k, t in raw[a:b] if k == 'i']
    return {
        'prologue': ins(0, h),
        'guard':    raw[gi][1],                    # the `@%pN bra EXIT` line
        'guard_pred': (gm.group(1) == '!', int(gm.group(2))),
        'pre_guard': ins(h + 1, gi),               # setp etc. before the test
        'body':     ins(gi + 1, e),                # excludes the back edge
        'epilogue': ins(gexit, len(raw)),
    }

# ---------------- SASS ----------------
SASS_INSN = re.compile(r'^\s*/\*([0-9a-f]+)\*/\s+(.*?);\s*$')
SASS_LBL  = re.compile(r'^(\.L_\w+):')
SASS_BRA  = re.compile(r'^(?:@(!?)P(\d+)\s+)?BRA\s+`\((\.L_\w+)\)$')

def sass_regions(path):
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
