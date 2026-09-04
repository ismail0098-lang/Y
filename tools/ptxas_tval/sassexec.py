"""Symbolic executor for a straight-line, predicated SASS basic block.

Every opcode form implemented here is one whose semantics was validated
against silicon in Phase A, EXCEPT the forms named in ASSUMED below.  An
opcode or operand form that is not implemented is a hard error -- never a
no-op and never a guess.  (Design rule: an unhandled node in a pass whose
output is a correctness claim must reject.)
"""
import re, sys
from z3 import *
import smem

W = 32
def bv(n): return BitVecVal(n, W)
ZERO = bv(0)

# param base for sm_89; the driver ABI puts the kernel's parameters here.
CBANK = {0x160:'A_lo', 0x164:'A_hi', 0x168:'B_lo', 0x16c:'B_hi',
         0x170:'O_lo', 0x174:'O_hi', 0x178:'N',
         0x28:'stackptr', 0x118:'gridc_lo', 0x11c:'gridc_hi'}

ASSUMED = set()   # forms whose semantics is not device-validated

INSN = re.compile(r'^\s*/\*([0-9a-f]+)\*/\s+(.*?);\s*$')
LABEL = re.compile(r'^(\.L_\w+):')

def u64(hi, lo): return Concat(hi, lo)

class Sass:
    def __init__(self, sym):
        self.R = {}; self.P = {}; self.UR = {}
        self.sym = sym                    # shared symbol table with the PTX side
        self.mem = sym['mem']
        self.alive = BoolVal(True)
        self.stores = []
        self.loads = []
        self.defs = []          # (pc, name, expr) for every register definition
        self.wide = []          # (pc, 33-bit value:carry) for accumulating insns
        self.pc = 0
        self.undef = 0
        self.count = 0
        self.forms = set()
        self.smem = sym.get('smem')       # see smem.py
        self.bar = sym.get('bar')
        self.smem_ops = []
        self.smem_snaps = []      # array contents ENTERING each barrier
        self.align_obs = []

    # ---- operand readers -------------------------------------------------
    def rd(self, o):
        o = o.strip().replace('.reuse','')
        if o.startswith('-') and not o.startswith('-0x'): return -self.rd(o[1:])
        if o.startswith('~'): return ~self.rd(o[1:])
        if o in ('RZ','URZ'): return ZERO
        m = re.fullmatch(r'R(\d+)(\.64)?', o)
        if m:
            i = int(m.group(1))
            if i not in self.R:
                self.undef += 1
                self.R[i] = BitVec(f'sass_undef_R{i}', W)
            return self.R[i]
        m = re.fullmatch(r'c\[0x0\]\[0x([0-9a-f]+)\]', o)
        if m:
            a = int(m.group(1), 16)
            cb = self.sym.get('cbank', CBANK)
            if a not in cb: raise Exception(f'unmodelled const bank slot 0x{a:x}')
            v = cb[a]
            return self.sym[v if isinstance(v,str) else f'{v[0]}_{v[1]}']
        m = re.fullmatch(r'UR(\d+)', o)
        if m: return self.UR[int(m.group(1))]
        m = re.fullmatch(r'(-?)0x([0-9a-f]+)', o)
        if m:
            v = int(m.group(2), 16)
            return bv(-v if m.group(1) else v)
        raise Exception(f'unmodelled operand {o!r}')

    def gaddr(self, o):
        """A 64-bit GLOBAL address operand: `[Rn.64]` or `[Rn.64+0xNN]`.

        Five arms parsed this inline and FOUR OF THEM DID NOT CHECK THE MATCH,
        so an addressing form the model does not know crashed with
        `'NoneType' object has no attribute 'group'` instead of refusing by
        name -- which in a corpus census is indistinguishable from a bug in the
        harness, and told nobody that `[Rn.64+0xNN]` was simply unsupported.
        The constant-offset form is ordinary (37 plain and 5 offset in
        `bn254_ntt4_fused` alone); it was never rejected on purpose, it was
        never reached.
        """
        m = re.fullmatch(r'\[R(\d+)\.64(?:\s*\+\s*(-?0x[0-9a-f]+|-?\d+))?\]', o.strip())
        if not m:
            raise Exception(f'unmodelled global addressing {o!r}  (refusing, not guessing)')
        ab = int(m.group(1))
        addr = u64(self.R[ab+1], self.R[ab])
        if m.group(2):
            addr = addr + BitVecVal(int(m.group(2), 0), 64)
        return simplify(addr)

    def saddr(self, o):
        """A SHARED address operand: `[R2.X16]`, `[R0.X16+0x400]`, `[R2]`, `[UR4]`.

        `.XN` is an address-SCALING modifier -- `[R2.X16]` is `R2 * 16`, not
        `R2`.  The PTX side reaches the same address by an explicit `shl.b64
        %rd, %rd, 4`, so dropping the scale does not fail loudly: it produces a
        DIFFERENT address that is still well-formed, and the validator would
        report a real kernel as a mismatch.  Refuse an unknown form instead.
        """
        m = re.fullmatch(r'\[(.*)\]', o.strip())
        if not m: raise Exception(f'unmodelled shared address {o!r}  (refusing, not guessing)')
        body = m.group(1).strip()
        off = BitVecVal(0, W)
        m2 = re.fullmatch(r'(.*?)\s*\+\s*(-?0x[0-9a-f]+|-?\d+)', body)
        if m2:
            body = m2.group(1).strip()
            off = bv(int(m2.group(2), 0))
        m3 = re.fullmatch(r'(R\d+|UR\d+|RZ|URZ)(?:\.X(\d+))?', body)
        if not m3: raise Exception(f'unmodelled shared address form {o!r}  (refusing, not guessing)')
        base = self.rd(m3.group(1))
        if m3.group(2):
            base = base * bv(int(m3.group(2)))
        return simplify(base + off)

    def pr(self, o):
        o = o.strip()
        neg = o.startswith('!')
        if neg: o = o[1:]
        if o == 'PT': v = BoolVal(True)
        else:
            i = int(o[1:])
            if i not in self.P:
                self.undef += 1
                self.P[i] = Bool(f'sass_undef_P{i}')
            v = self.P[i]
        return Not(v) if neg else v

    # ---- writers (predicated) -------------------------------------------
    def wr(self, o, val, g):
        o = o.strip()
        if o == 'RZ': return
        i = int(o[1:])
        old = self.R.get(i)
        if old is None:
            self.undef += 1
            old = BitVec(f'sass_undef_R{i}', W)
        self.R[i] = simplify(val if is_true(g) else If(g, val, old))
        self.defs.append((self.pc, o, self.R[i]))

    def widen(self, val, carry):
        self.wide.append((self.pc,
                          simplify(Concat(If(carry, BitVecVal(1,1), BitVecVal(0,1)), val)),
                          val, carry))

    def wp(self, o, val, g):
        o = o.strip()
        if o in ('PT','!PT'): return
        i = int(o[1:])
        old = self.P.get(i)
        if old is None:
            self.undef += 1
            old = Bool(f'sass_undef_P{i}')
        self.P[i] = simplify(val if is_true(g) else If(g, val, old))

    # ---- primitives ------------------------------------------------------
    def add3(self, a, b, c, cin1=None, cin2=None, ones=0):
        """three-input 32-bit add with up to two carry-ins; returns (sum, carry-out).
        `ones` counts source operands written as `-R`, each of which was passed
        in already inverted and contributes one to the carry chain."""
        w = W + 3
        e = lambda x: ZeroExt(3, x)
        s = e(a) + e(b) + e(c)
        if ones: s = s + BitVecVal(ones, w)
        for ci in (cin1, cin2):
            if ci is not None: s = s + If(ci, BitVecVal(1, w), BitVecVal(0, w))
        return Extract(W-1, 0, s), (LShR(s, W) != 0)

    def src3(self, o):
        """an adder source: returns (value, extra ones for the carry chain)"""
        o = o.strip().replace('.reuse','')
        if o.startswith('-') and not o.startswith('-0x'): return ~self.rd(o[1:]), 1
        return self.rd(o), 0

    def mul_lo(self, a, b):
        f = self.sym.get('mul')
        return f('lo', a, b) if f else a * b
    def mul_hi(self, a, b):
        f = self.sym.get('mul')
        return f('hi', a, b) if f else Extract(2*W-1, W, ZeroExt(W, a) * ZeroExt(W, b))

    def pair(self, o):
        """a 64-bit operand held in the register pair starting at o (RZ is 0)"""
        o = o.strip().replace('.reuse','')
        if o == 'RZ': return BitVecVal(0, 64)
        m = re.fullmatch(r'R(\d+)', o)
        if not m: raise Exception(f'unmodelled wide operand {o!r}')
        i = int(m.group(1))
        return Concat(self.rd(f'R{i+1}'), self.rd(f'R{i}'))

    def mul_hi_wide(self, a, b, caddr):
        """high word of a*b + {Rc,Rc+1}, and the carry out of that 64-bit add"""
        prod = ZeroExt(1, ZeroExt(W, a) * ZeroExt(W, b))
        s = prod + ZeroExt(1, self.pair(caddr))
        return Extract(2*W-1, W, Extract(2*W-1, 0, s)), (Extract(2*W, 2*W, s) == BitVecVal(1,1))

    def lop3(self, a, b, c, lut):
        r = ZERO
        for i in range(8):
            if (lut >> i) & 1:
                ma = a if (i >> 2) & 1 else ~a
                mb = b if (i >> 1) & 1 else ~b
                mc = c if (i >> 0) & 1 else ~c
                r = r | (ma & mb & mc)
        return r

    # ---- the interpreter -------------------------------------------------
    def arrive(self, addr):
        """Re-merge the path condition at a branch target.

        A forward branch splits the path; the instructions it skips already
        executed under `alive AND NOT taken`, so every write they made is
        guarded and the registers merge themselves.  What has to be restored
        is the path condition: at the join it is the disjunction of the two
        incoming edges.  Stating it that way (rather than restoring the saved
        value) keeps it correct when the skipped region contains an EXIT.
        """
        while self.joins and self.joins[-1][0] == addr:
            _, taken = self.joins.pop()
            self.alive = simplify(Or(taken, self.alive))

    def step(self, body, addr=None):
        """Dispatch one instruction.

        AN ARITY MISTAKE MUST NAME ITSELF.  Arms index `ops[k]` at fixed
        positions, so an instruction with fewer operands than its arm expects
        raised a bare `IndexError: list index out of range` -- fail-closed, and
        useless: in a corpus census that is indistinguishable from a bug in the
        harness, and it told nobody that a second `LEA` arity existed.  Wrapping
        the dispatch closes the whole class at one point rather than adding a
        length check to forty arms, for the same reason `gaddr` replaced five
        inline address parsers."""
        try:
            return self._step(body, addr)
        except IndexError:
            t = body.split(';')[0].strip()
            raise Exception(f'UNMODELLED SASS OPERAND COUNT in {t!r}  '
                            f'(an arm indexed past the operand list; refusing, not guessing)')

    def _step(self, body, addr=None):
        pred = None
        m = re.match(r'^(@!?U?P\w+)\s+(.*)$', body)
        if m:
            pred, body = m.group(1), m.group(2)
        self.pc += 1
        parts = body.split(None, 1)
        opc = parts[0]
        ops = [o.strip() for o in parts[1].split(',')] if len(parts) > 1 else []
        g = self.alive if pred is None else And(self.alive, self.pr(pred[1:]))
        self.count += 1
        self.forms.add(opc)
        R, rd = self.rd, self.rd

        if opc in ('NOP',):
            pass
        elif opc == 'UMOV':
            self.UR[int(ops[0][2:])] = rd(ops[1])
        elif opc in ('MOV', 'MOV32I'):
            self.wr(ops[0], rd(ops[1]), g)
        elif opc == 'EXIT':
            # execution stops where the guard holds
            self.alive = And(self.alive, Not(g))
        elif opc == 'BRA':
            m = re.search(r'`\(\.L_(\w+)\)', body)
            if not m: raise Exception(f'unmodelled BRA form {body!r}  (refusing, not guessing)')
            tgt = self.labels.get(m.group(1))
            if tgt is None: raise Exception(f'BRA to unknown label {m.group(1)!r}')
            if tgt == addr:
                # nvcc emits `.L: BRA .L` after EXIT as a trap.  It is dead
                # code -- but only if it is genuinely unreachable, so check
                # rather than assume: a reachable self-branch is a loop.
                if not is_false(simplify(self.alive)):
                    raise Exception('self-branch reached under a satisfiable path condition '
                                    '-- that is a loop, not the trailing trap  (refusing, not guessing)')
            elif tgt < addr:
                raise Exception('backward BRA -- this kernel has a loop, which needs an '
                                'invariant rather than path merging  (refusing, not guessing)')
            else:
                self.joins.append((tgt, g))
                self.alive = simplify(And(self.alive, Not(g)))
        elif opc in ('FMUL','FADD','FFMA','FSUB'):
            n = 3 if opc == 'FFMA' else 2
            self.wr(ops[0], self.sym['fp'](opc, *[rd(o) for o in ops[1:1+n]], side='sass'), g)
        elif opc.startswith('SHF.'):
            # funnel shift: {Rc:Ra} shifted, .HI takes the upper word.
            f = opc.split('.')
            if len(f) != 4 or f[3] != 'HI' or f[1] not in ('L','R') or f[2] not in ('U32','S32'):
                raise Exception(f'UNMODELLED SASS OPCODE {opc!r}  (refusing, not guessing)')
            wide = Concat(rd(ops[3]), rd(ops[1]))
            n64  = ZeroExt(32, rd(ops[2]))
            v = (wide << n64) if f[1] == 'L' else (
                 LShR(wide, n64) if f[2] == 'U32' else (wide >> n64))
            self.wr(ops[0], Extract(63, 32, v), g)
        elif opc in ('BSSY', 'BSYNC'):
            # Warp reconvergence.  These manage the divergence stack; they move
            # no value into any register this model reads.  That is only sound
            # while the kernel cannot observe which lanes are converged, so
            # run_sass REFUSES a kernel containing any cross-thread operation
            # rather than letting this arm quietly cover one.
            ASSUMED.add(f'{opc} = no-op on the value domain (checked: kernel has no cross-thread op)')
        elif opc == 'S2R':
            src = ops[1]
            key = src[3:].lower().replace('.','_') if src.startswith('SR_') else None
            if key not in self.sym: raise Exception(f'unmodelled special register {src}')
            self.wr(ops[0], self.sym[key], g)
        elif opc == 'ULDC':
            m = re.fullmatch(r'c\[0x0\]\[0x([0-9a-f]+)\]', ops[1])
            if not m: raise Exception(f'unmodelled ULDC source {ops[1]!r}  (refusing, not guessing)')
            self.UR[int(ops[0][2:])] = rd(ops[1])
        elif opc.startswith('USHF.'):
            # the uniform-datapath funnel shift; identical semantics to SHF,
            # different register file.  Shares the SHF code rather than copying
            # it -- two spellings of one operation drift.
            f = opc.split('.')
            if len(f) != 3 or f[1] not in ('L','R') or f[2] not in ('U32','S32'):
                raise Exception(f'UNMODELLED SASS OPCODE {opc!r}  (refusing, not guessing)')
            wide = Concat(rd(ops[3]), rd(ops[1]))
            n64  = ZeroExt(32, rd(ops[2]))
            v = (wide << n64) if f[1] == 'L' else (
                 LShR(wide, n64) if f[2] == 'U32' else (wide >> n64))
            self.UR[int(ops[0][2:])] = simplify(Extract(31, 0, v))
        elif opc.startswith('STS'):
            n = {'STS': 1, 'STS.64': 2, 'STS.128': 4}.get(opc)
            if n is None: smem.refuse_subword(opc)
            addr = self.saddr(ops[0])
            self.align_obs.append(smem.require_aligned(addr))
            self.smem_ops.append(('st', addr, g))
            vb = int(ops[1][1:])
            for k in range(n):
                idx = smem.word(addr + bv(4*k))
                nxt = Store(self.smem, idx, rd(f'R{vb+k}'))
                self.smem = simplify(If(g, nxt, self.smem) if not is_true(g) else nxt)
        elif opc.startswith('LDS'):
            n = {'LDS': 1, 'LDS.64': 2, 'LDS.128': 4}.get(opc)
            if n is None: smem.refuse_subword(opc)
            addr = self.saddr(ops[1])
            self.align_obs.append(smem.require_aligned(addr))
            self.smem_ops.append(('ld', addr, g))
            db = int(ops[0][1:])
            for k in range(n):
                self.wr(f'R{db+k}', Select(self.smem, smem.word(addr + bv(4*k))), g)
        elif opc.startswith('BAR.SYNC'):
            # NOT a no-op -- see smem.py.  A predicated barrier is a program the
            # hardware does not admit either (a partial arrival hangs).
            if not is_true(g):
                raise Exception('BAR.SYNC under a predicate  (refusing, not guessing)')
            self.smem_snaps.append(self.smem)
            self.smem = self.bar.apply(self.smem, 'sass')
        elif opc == 'ULDC.64':
            m = re.fullmatch(r'c\[0x0\]\[0x([0-9a-f]+)\]', ops[1])
            if not m: raise Exception(f'unmodelled ULDC.64 source {ops[1]!r}  (refusing, not guessing)')
            a = int(m.group(1), 16)
            i = int(ops[0][2:])
            cb = self.sym.get('cbank', CBANK)
            def g2(x):
                v = cb.get(x)
                return self.sym[v if isinstance(v,str) else f'{v[0]}_{v[1]}'] if v is not None else BitVecVal(0,W)
            self.UR[i] = g2(a); self.UR[i+1] = g2(a+4)
        elif opc in ('IMAD', 'IMAD.MOV.U32', 'IMAD.SHL.U32', 'IMAD.U32', 'IMAD.IADD'):
            if opc == 'IMAD.SHL.U32': ASSUMED.add('IMAD.SHL.U32 = IMAD (multiply pipe shift)')
            self.wr(ops[0], self.mul_lo(rd(ops[1]), rd(ops[2])) + rd(ops[3]), g)
        elif opc == 'IMAD.X':
            # d = lo(a*b) + c + P
            s, _ = self.add3(self.mul_lo(rd(ops[1]), rd(ops[2])), rd(ops[3]), ZERO, self.pr(ops[4]))
            self.wr(ops[0], s, g)
        elif opc == 'IMAD.HI.U32':
            # The addend is a 64-BIT REGISTER PAIR {Rc, Rc+1} and the result is
            # the HIGH word of a*b + that pair; the predicate is the carry out
            # of the 64-bit addition.  Reading it as a 32-bit addend gives the
            # right answer only when the upper half happens to be zero -- and
            # ptxas puts PTX's `mad.hi.cc.u32` addend in the UPPER half, with
            # zero below, so the 32-bit reading is wrong on every such kernel.
            if len(ops) == 4:      # d, a, b, c
                s, _ = self.mul_hi_wide(rd(ops[1]), rd(ops[2]), ops[3])
                self.wr(ops[0], s, g)
            elif len(ops) == 5:    # d, Pout, a, b, c
                s, co = self.mul_hi_wide(rd(ops[2]), rd(ops[3]), ops[4])
                self.wr(ops[0], s, g); self.wp(ops[1], co, g); self.widen(s, co)
            else: raise Exception(f'unmodelled IMAD.HI.U32 arity {len(ops)}')
        elif opc in ('IMAD.WIDE.U32','IMAD.WIDE'):
            # {Rd+1,Rd} = a*b + {Rc+1,Rc}.  The full-width sibling of
            # IMAD.HI.U32, and the addend is a 64-bit REGISTER PAIR for the same
            # reason -- that correction is what made the partial sums correspond.
            # The product is built from the SHARED multiply primitive so that
            # under the `wide` posing both halves are extracts of one MUL64 node
            # and the concat folds back to it.
            sgn = '.U32' not in opc
            if sgn: ASSUMED.add('IMAD.WIDE (signed) = 64-bit signed product + pair')
            a, b = rd(ops[1]), rd(ops[2])
            prod = Concat(self.mul_hi(a, b), self.mul_lo(a, b))
            res = simplify(prod + self.pair(ops[3]))
            d = int(ops[0][1:])
            self.wr(f'R{d}',   Extract(W-1, 0, res), g)
            self.wr(f'R{d+1}', Extract(2*W-1, W, res), g)
        elif opc == 'IADD3':
            if len(ops) == 5:      # d, Pout, a, b, c
                (x,n1),(y,n2),(z,n3) = (self.src3(ops[2]), self.src3(ops[3]), self.src3(ops[4]))
                s, co = self.add3(x, y, z, ones=n1+n2+n3)
                self.wr(ops[0], s, g); self.wp(ops[1], co, g); self.widen(s, co)
            elif len(ops) == 4:    # d, a, b, c   (carry-out discarded)
                (x,n1),(y,n2),(z,n3) = (self.src3(ops[1]), self.src3(ops[2]), self.src3(ops[3]))
                s, _ = self.add3(x, y, z, ones=n1+n2+n3)
                self.wr(ops[0], s, g)
            else: raise Exception(f'unmodelled IADD3 arity {len(ops)}')
        elif opc == 'IADD3.X':
            if len(ops) == 7:      # d, Pout, a, b, c, Pin1, Pin2
                (x,n1),(y,n2),(z,n3) = (self.src3(ops[2]), self.src3(ops[3]), self.src3(ops[4]))
                s, co = self.add3(x, y, z, self.pr(ops[5]), self.pr(ops[6]), ones=n1+n2+n3)
                self.wr(ops[0], s, g); self.wp(ops[1], co, g); self.widen(s, co)
            elif len(ops) == 6:    # d, a, b, c, Pin1, Pin2   (no carry-out slot)
                (x,n1),(y,n2),(z,n3) = (self.src3(ops[1]), self.src3(ops[2]), self.src3(ops[3]))
                s, _ = self.add3(x, y, z, self.pr(ops[4]), self.pr(ops[5]), ones=n1+n2+n3)
                self.wr(ops[0], s, g)
            else: raise Exception(f'unmodelled IADD3.X arity {len(ops)}')
        elif opc == 'SEL':
            self.wr(ops[0], If(self.pr(ops[3]), rd(ops[1]), rd(ops[2])), g)
        elif opc == 'LOP3.LUT':
            lut = int(ops[4], 16)
            self.wr(ops[0], self.lop3(rd(ops[1]), rd(ops[2]), rd(ops[3]), lut), g)
        elif opc == 'LEA':
            # TWO ARITIES, and only one was modelled:
            #   d, Pout, a, C, shift   lo(C) + (a << shift), carry out
            #   d,       a, C, shift   the same sum with no carry consumer
            # The 4-operand form indexed ops[4] and crashed with IndexError
            # rather than refusing -- see the `step` wrapper for why that class
            # is now closed at one point instead of arm by arm.
            if len(ops) == 5:
                sh = int(ops[4], 16)
                v, co = self.add3(rd(ops[3]), rd(ops[2]) << bv(sh), ZERO)
                self.wr(ops[0], v, g); self.wp(ops[1], co, g)
            elif len(ops) == 4:
                sh = int(ops[3], 16)
                v, _ = self.add3(rd(ops[2]), rd(ops[1]) << bv(sh), ZERO)
                self.wr(ops[0], v, g)
            else:
                raise Exception(f'unmodelled LEA arity {len(ops)}  (refusing, not guessing)')
        elif opc == 'LEA.HI.X':
            # d, a, C, RZ, shift, Pin :  hi(C) + (a >> (32-shift)) + Pin
            sh = int(ops[4], 16)
            hi = LShR(rd(ops[1]), bv(W - sh))
            s, _ = self.add3(rd(ops[2]), hi, rd(ops[3]), self.pr(ops[5]))
            self.wr(ops[0], s, g)
        elif opc.startswith('ISETP.'):
            # ISETP.<cmp>[.U32].<comb>  Pd, Pd2, Ra, Rb, Pin
            # Pd2 is a second predicate output this model does not track, so a
            # use of it is refused rather than dropped; every kernel here writes
            # PT (discard) there.
            f = opc.split('.')
            comb = f[-1]
            cmp_ = f[1]
            uns  = 'U32' in f
            CMPS = {'LT': (ULT, lambda x,y: x<y), 'LE': (ULE, lambda x,y: x<=y),
                    'GT': (UGT, lambda x,y: x>y), 'GE': (UGE, lambda x,y: x>=y),
                    'EQ': (lambda x,y: x==y, lambda x,y: x==y),
                    'NE': (lambda x,y: x!=y, lambda x,y: x!=y)}
            COMBS = {'AND': And, 'OR': Or, 'XOR': Xor}
            if cmp_ not in CMPS or comb not in COMBS:
                raise Exception(f'UNMODELLED SASS OPCODE {opc!r}  (refusing, not guessing)')
            if len(f) > 4 or (len(f)==4 and not uns):
                raise Exception(f'unmodelled ISETP qualifier in {opc!r}  (refusing, not guessing)')
            if ops[1] != 'PT':
                raise Exception(f'ISETP writes a second predicate {ops[1]!r}, which this model '
                                f'does not track  (refusing, not guessing)')
            rel = CMPS[cmp_][0 if uns else 1](rd(ops[2]), rd(ops[3]))
            self.wp(ops[0], COMBS[comb](rel, self.pr(ops[4])), g)
        elif opc == 'LDG.E.128':
            base = int(ops[0][1:])
            addr = self.gaddr(ops[1])
            i = len(self.loads); self.loads.append((addr, g))
            for k in range(4):
                v = (self.sym['abstract'](i, k) if 'abstract' in self.sym
                     else Select(self.mem, addr + BitVecVal(4*k, 64)))
                self.wr(f'R{base+k}', v, g)
        elif opc in ('LDG.E','LDG.E.U32','LDG.E.128.CONSTANT','LDG.E.CONSTANT'):
            addr = self.gaddr(ops[1])
            i = len(self.loads); self.loads.append((addr, g))
            nw = 4 if '128' in opc else 1
            base = int(ops[0][1:])
            for k in range(nw):
                v = (self.sym['abstract'](i,k) if 'abstract' in self.sym
                     else Select(self.mem, addr + BitVecVal(4*k,64)))
                self.wr(f'R{base+k}', v, g)
        elif opc in ('LDG.E.S8','LDG.E.U8','LDG.E.S16','LDG.E.U16'):
            # same word-per-byte-address convention as ptxexec's ld.global.s8;
            # the two must agree or nothing downstream means anything
            addr = self.gaddr(ops[1])
            i = len(self.loads); self.loads.append((addr, g))
            w = (self.sym['abstract'](i,0) if 'abstract' in self.sym
                 else Select(self.mem, addr))
            nb = 8 if opc.endswith('8') else 16
            byte = Extract(nb-1, 0, w)
            self.wr(ops[0], (SignExt(W-nb, byte) if '.S' in opc else ZeroExt(W-nb, byte)), g)
        elif opc == 'STG.E.64':
            addr = self.gaddr(ops[0]); vb = int(ops[1][1:])
            # split exactly as ptxexec's st.global.u64 does: lo at addr, hi at +4
            self.stores.append((addr, self.rd(f'R{vb}'), g))
            self.stores.append((addr + BitVecVal(4, 64), self.rd(f'R{vb+1}'), g))
        elif opc in ('STG.E',):
            addr = self.gaddr(ops[0])
            self.stores.append((addr, self.rd(ops[1]), g))
        elif opc == 'STG.E.128':
            addr = self.gaddr(ops[0]); vb = int(ops[1][1:])
            for k in range(4):
                self.stores.append((addr + BitVecVal(4*k, 64), self.rd(f'R{vb+k}'), g))
        else:
            raise Exception(f'UNMODELLED SASS OPCODE {opc!r}  (refusing, not guessing)')

CROSS_THREAD = re.compile(r'\b(BAR\.|MEMBAR|SHFL|VOTE|LDS|STS|LDSM|ATOM|RED|MATCH)')

def run_insns(insns, sym, name='region', seed=None):
    """Run an explicit list of (address, text) SASS instructions.

    Same purpose as ptxexec.run_lines: execute ONE REGION in a fresh state so
    that its live-ins materialise as `sass_undef_RN` / `sass_undef_PN`.

    Forward branches WITHIN the region still merge through arrive(); a branch
    whose target is outside the region is a region exit and is the caller's
    business, so it is refused here rather than silently dropped."""
    st = Sass(sym)
    st.labels = {}
    st.joins = []
    # SEEDING.  A region's live-ins are normally fresh `sass_undef_*` symbols.
    # Seeding lets the caller run this region in ANOTHER program's vocabulary --
    # which the loop validator needs, because rewriting the result afterwards
    # cannot undo a decision taken while it was being built: the multiply
    # primitive canonicalises its operands by z3 node id AT CONSTRUCTION, so two
    # terms that only become equal after substitution can already have their
    # operands in opposite orders, and congruence closure will not relate them.
    if seed:
        for k, v in seed.items():
            (st.P if isinstance(k, str) else st.R)[int(str(k).lstrip('P')) if isinstance(k, str) else k] = v
    lo = min(a for a, _ in insns); hi = max(a for a, _ in insns)
    for a, t in insns:
        m = re.search(r'BRA\s+`\((\.L_\w+)\)', t)
        if m:
            raise Exception(f'{name}: branch at 0x{a:x} inside a region  '
                            f'(refusing, not guessing)')
        st.arrive(a)
        st.step(t, a)
    if st.joins:
        raise Exception(f'{name}: {len(st.joins)} unreached branch target(s)')
    return st


def run_sass(path, sym):
    text = open(path).read()
    st = Sass(sym); looks = 0
    st.labels = {m.group(1): int(m.group(2), 16)
                 for m in re.finditer(r'\.L_(\w+):\s*\n\s*/\*([0-9a-f]+)\*/', text)}
    st.joins = []
    if re.search(r'\bBSSY\b|\bBSYNC\b', text) and CROSS_THREAD.search(text):
        raise Exception('kernel has both warp reconvergence and a cross-thread operation; '
                        'the no-op reading of BSSY/BSYNC is not sound here  (refusing, not guessing)')
    for line in text.splitlines():
        if '/*' in line and '*/' in line and line.strip().endswith(';'): looks += 1
        m = INSN.match(line)
        if not m: continue
        addr = int(m.group(1), 16)
        st.arrive(addr)
        st.step(m.group(2).strip(), addr)
    if st.joins:
        raise Exception(f'{len(st.joins)} branch target(s) never reached -- the CFG is not '
                        f'what the linear scan assumed  (refusing, not guessing)')
    # A parser that silently drops instructions produces a wrong model that
    # still runs.  (The address field was matched at a fixed four hex digits,
    # which truncates every kernel larger than 64 KB of code.)
    if st.count != looks:
        raise Exception(f'parsed {st.count} of {looks} instruction lines -- parser is dropping instructions')
    return st

if __name__ == '__main__':
    sym = {k: BitVec(k, W) for k in
           ('A_lo','A_hi','B_lo','B_hi','O_lo','O_hi','N','stackptr',
            'gridc_lo','gridc_hi','ctaid_x','tid_x')}
    sym['mem'] = Array('mem', BitVecSort(64), BitVecSort(32))
    st = run_sass(sys.argv[1], sym)
    print(f'executed {st.count} instructions, {len(st.forms)} opcodes')
    print(f'stores: {len(st.stores)}   undefined reads materialised: {st.undef}')
    print(f'assumed semantics: {sorted(ASSUMED) or "none"}')
