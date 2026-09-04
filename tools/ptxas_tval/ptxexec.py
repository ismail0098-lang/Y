"""Symbolic executor for the straight-line PTX this kernel is compiled from.

Shares the SAME multiplier primitive as the SASS executor -- both sides
compute 32x32 products and take the halves separately -- which is what keeps
the equivalence query in the tractable regime (Phase A's representation rule).

An unmodelled opcode is a hard error.
"""
import re, sys
from z3 import *
import fpmode
import smem

W = 32
def bv(n): return BitVecVal(n, W)
ZERO = bv(0)
# parameter names are read from the kernel header, not hardcoded

class Ptx:
    def __init__(self, sym):
        self.r = {}; self.rd = {}; self.p = {}; self.f = {}
        self.cc = BoolVal(False)          # the PTX carry flag
        self.sym = sym; self.mem = sym['mem']
        self.stores = []; self.loads = []; self.defs = []; self.wide = []; self.pc = 0; self.count = 0; self.ops = set(); self.undef = 0
        # Shared memory is STATE, not a trace.  The global loads/stores above are
        # a log the validator matches by address permutation, which works only
        # because a kernel never reads back what it wrote to global memory in the
        # same launch.  A shared roundtrip is exactly that read-back, so it needs
        # an array.  See smem.py for the barrier model.
        self.smem = sym.get('smem')
        self.bar = sym.get('bar')
        self.smem_layout = sym.get('smem_layout', {})
        self.smem_ops = []
        self.smem_snaps = []      # array contents ENTERING each barrier          # (kind, byte_addr, guard) for diagnostics
        self.align_obs = []         # 4-byte alignment obligations

    def R(self, o):
        o = o.strip()
        m = re.fullmatch(r'%r(\d+)', o)
        if m:
            i = int(m.group(1))
            if i not in self.r:
                self.undef += 1; self.r[i] = BitVec(f'ptx_undef_r{i}', W)
            return self.r[i]
        m = re.fullmatch(r'(-?)(\d+)', o)
        if m: return bv(-int(m.group(2)) if m.group(1) else int(m.group(2)))
        m = re.fullmatch(r'(-?)0[xX]([0-9a-fA-F]+)', o)
        if m: return bv(-int(m.group(2),16) if m.group(1) else int(m.group(2),16))
        # A `.shared` symbol can be read into a 32-bit register as well as a
        # 64-bit one -- the GEMMs do `mov.u32 %r26, smem_pipeline_...` because
        # the shared window IS 32-bit addressed.  Wiring only D() covered the
        # one kernel written the other way; enumerate the SITES.
        if o in self.smem_layout:
            return BitVecVal(self.smem_layout[o], W)
        raise Exception(f'unmodelled 32-bit operand {o!r}')

    def F(self, o):
        """A float operand: %fN, or a PTX float literal 0f<8 hex> / a decimal.

        Floats are carried as their 32-bit PATTERN, so they flow through the
        same register machinery as everything else; only the operations are
        abstract.  %f5 and %r5 are DIFFERENT registers, hence the separate
        file -- the writer keys on the number alone and they would collide.
        """
        o = o.strip()
        m = re.fullmatch(r'%f(\d+)', o)
        if m:
            i = int(m.group(1))
            if i not in self.f:
                self.undef += 1; self.f[i] = BitVec(f'ptx_undef_f{i}', W)
            return self.f[i]
        m = re.fullmatch(r'0[fF]([0-9a-fA-F]{8})', o)
        if m: return BitVecVal(int(m.group(1), 16), W)
        import struct
        m = re.fullmatch(r'-?\d+\.\d*', o)
        if m: return BitVecVal(struct.unpack('>I', struct.pack('>f', float(o)))[0], W)
        raise Exception(f'unmodelled float operand {o!r}')

    def wf(self, o, v, g):
        i = int(o.strip()[2:])
        old = self.f.get(i)
        if old is None: self.undef += 1; old = BitVec(f'ptx_undef_f{i}', W)
        self.f[i] = simplify(If(g, v, old) if not is_true(g) else v)
        self.defs.append((self.pc, o.strip(), self.f[i]))

    def D(self, o):
        o = o.strip()
        m = re.fullmatch(r'%rd(\d+)', o)
        if m:
            i = int(m.group(1))
            if i not in self.rd:
                self.undef += 1; self.rd[i] = BitVec(f'ptx_undef_rd{i}', 64)
            return self.rd[i]
        # A `.shared` symbol is its BASE OFFSET in the shared window.  SASS
        # addresses that window from 0, so the base is what makes the two sides'
        # addresses comparable at all; smem.layout refuses more than one array
        # rather than guess the second base.
        if o in self.smem_layout:
            return BitVecVal(self.smem_layout[o], 64)
        m = re.fullmatch(r'(-?)(\d+)', o)
        if m: return BitVecVal(-int(m.group(2)) if m.group(1) else int(m.group(2)), 64)
        raise Exception(f'unmodelled 64-bit operand {o!r}')

    def SA(self, o):
        """A shared-memory address operand, from either register file.

        `[%rd7]` and `[%r102]` both occur; the shared window is 32-bit, so both
        reduce to 32 bits here rather than at the point of use."""
        o = o.strip()
        return self.D(o) if o.startswith('%rd') else self.R(o)

    def P(self, o):
        o = o.strip(); neg = o.startswith('!')
        if neg: o = o[1:]
        i = int(o[2:])
        if i not in self.p:
            self.undef += 1; self.p[i] = Bool(f'ptx_undef_p{i}')
        return Not(self.p[i]) if neg else self.p[i]

    def wr(self, o, v, g):
        i = int(o.strip()[2:])
        old = self.r.get(i)
        if old is None: self.undef += 1; old = BitVec(f'ptx_undef_r{i}', W)
        self.r[i] = simplify(If(g, v, old) if not is_true(g) else v)
        self.defs.append((self.pc, o.strip(), self.r[i]))
    def wd(self, o, v, g):
        i = int(o.strip()[3:])
        old = self.rd.get(i)
        if old is None: self.undef += 1; old = BitVec(f'ptx_undef_rd{i}', 64)
        self.rd[i] = simplify(If(g, v, old) if not is_true(g) else v)
    def widen(self, val, carry):
        self.wide.append((self.pc,
                          simplify(Concat(If(carry, BitVecVal(1,1), BitVecVal(0,1)), val)),
                          val, carry))

    def wp(self, o, v, g):
        i = int(o.strip()[2:])
        old = self.p.get(i)
        if old is None: self.undef += 1; old = Bool(f'ptx_undef_p{i}')
        self.p[i] = simplify(If(g, v, old) if not is_true(g) else v)

    def addc(self, a, b, cin):
        """32-bit add with optional carry-in; returns (sum, carry-out)"""
        e = lambda x: ZeroExt(1, x)
        s = e(a) + e(b)
        if cin is not None: s = s + If(cin, BitVecVal(1, 33), BitVecVal(0, 33))
        return Extract(31,0,s), (Extract(32,32,s) == BitVecVal(1,1))
    def subb(self, a, b, bin_):
        e = lambda x: ZeroExt(1, x)
        s = e(a) - e(b)
        if bin_ is not None: s = s - If(bin_, BitVecVal(0,33), BitVecVal(1,33))
        return Extract(31,0,s), (Extract(32,32,s) == BitVecVal(0,1))  # 1 => no borrow
    def lo(self, a, b):
        f = self.sym.get('mul')
        return f('lo', a, b) if f else a * b
    def hi(self, a, b):
        f = self.sym.get('mul')
        return f('hi', a, b) if f else Extract(2*W-1, W, ZeroExt(W,a) * ZeroExt(W,b))

    def step(self, line):
        pred = None
        m = re.match(r'^@(!?%p\d+)\s+(.*)$', line)
        if m: pred, line = m.group(1), m.group(2)
        g = BoolVal(True) if pred is None else self.P(pred)
        self.pc += 1
        parts = line.split(None, 1)
        op = parts[0]
        rest = parts[1] if len(parts) > 1 else ''
        self.count += 1; self.ops.add(op)
        # split on commas at depth 0 (braces group vector operands)
        ops, depth, cur = [], 0, ''
        for ch in rest:
            if ch in '{[': depth += 1
            if ch in '}]': depth -= 1
            if ch == ',' and depth == 0: ops.append(cur); cur = ''
            else: cur += ch
        if cur.strip(): ops.append(cur)
        ops = [o.strip() for o in ops]

        if op.startswith('ld.param.'):
            nm = re.fullmatch(r'\[(\w+)\]', ops[1]).group(1)
            if op.endswith('64'):
                self.wd(ops[0], Concat(self.sym[nm+'_hi'], self.sym[nm+'_lo']), g)
            else:
                self.wr(ops[0], self.sym[nm+'_lo'], g)
        elif op == 'mov.u32':
            src = ops[1]
            if src.startswith('%') and not re.fullmatch(r'%r\d+', src):
                key = src[1:].replace('.','_')
                if key not in self.sym: raise Exception(f'unmodelled special register {src}')
                v = self.sym[key]
            else: v = self.R(src)
            self.wr(ops[0], v, g)
        elif op in ('mul.lo.s32','mul.lo.u32'):
            self.wr(ops[0], self.lo(self.R(ops[1]), self.R(ops[2])), g)
        elif op == 'add.s32':
            self.wr(ops[0], self.R(ops[1]) + self.R(ops[2]), g)
        elif op == 'and.b32':  self.wr(ops[0], self.R(ops[1]) & self.R(ops[2]), g)
        elif op == 'or.b32':   self.wr(ops[0], self.R(ops[1]) | self.R(ops[2]), g)
        elif op == 'xor.b32':  self.wr(ops[0], self.R(ops[1]) ^ self.R(ops[2]), g)
        elif op == 'setp.lt.u32':
            self.wp(ops[0], ULT(self.R(ops[1]), self.R(ops[2])), g)
        elif op == 'and.pred':
            self.wp(ops[0], And(self.P(ops[1]), self.P(ops[2])), g)
        elif op == 'cvt.u64.u32':
            self.wd(ops[0], ZeroExt(32, self.R(ops[1])), g)
        elif op == 'shl.b64':
            self.wd(ops[0], self.D(ops[1]) << BitVecVal(int(ops[2]), 64), g)
        elif op == 'add.u64':
            self.wd(ops[0], self.D(ops[1]) + self.D(ops[2]), g)
        elif op in ('st.shared.v4.u32','st.shared.v4.b32','st.shared.v4.f32'):
            addr = self.SA(re.fullmatch(r'\[(.*)\]', ops[0]).group(1))
            srcs = re.fullmatch(r'\{(.*)\}', ops[1]).group(1).split(',')
            rd = self.F if op.endswith('f32') else self.R
            self.align_obs.append(smem.require_aligned(addr))
            self.smem_ops.append(('st', addr, g))
            for k, sname in enumerate(srcs):
                idx = smem.word(addr + BitVecVal(4*k, addr.size()))
                nxt = Store(self.smem, idx, rd(sname))
                self.smem = simplify(If(g, nxt, self.smem) if not is_true(g) else nxt)
        elif op in ('ld.shared.v4.u32','ld.shared.v4.b32','ld.shared.v4.f32'):
            dsts = re.fullmatch(r'\{(.*)\}', ops[0]).group(1).split(',')
            addr = self.SA(re.fullmatch(r'\[(.*)\]', ops[1]).group(1))
            self.align_obs.append(smem.require_aligned(addr))
            self.smem_ops.append(('ld', addr, g))
            for k, d in enumerate(dsts):
                idx = smem.word(addr + BitVecVal(4*k, addr.size()))
                self.wr(d, Select(self.smem, idx), g)
        elif op in ('st.shared.u32','st.shared.s32','st.shared.b32','st.shared.f32'):
            addr = self.SA(re.fullmatch(r'\[(.*)\]', ops[0]).group(1))
            rd = self.F if op.endswith('f32') else self.R
            self.align_obs.append(smem.require_aligned(addr))
            self.smem_ops.append(('st', addr, g))
            idx = smem.word(addr)
            nxt = Store(self.smem, idx, rd(ops[1]))
            self.smem = simplify(If(g, nxt, self.smem) if not is_true(g) else nxt)
        elif op in ('ld.shared.u32','ld.shared.s32','ld.shared.b32','ld.shared.f32'):
            addr = self.SA(re.fullmatch(r'\[(.*)\]', ops[1]).group(1))
            self.align_obs.append(smem.require_aligned(addr))
            self.smem_ops.append(('ld', addr, g))
            self.wr(ops[0], Select(self.smem, smem.word(addr)), g)
        elif op.startswith(('ld.shared.','st.shared.')):
            smem.refuse_subword(op)
        elif op.startswith('bar.sync') or op.startswith('barrier.sync'):
            # NOT a no-op.  See smem.py: a no-op barrier validates a kernel whose
            # shared store ptxas moved across it, which is the main thing here
            # worth catching.  A barrier under a guard is a program neither this
            # model nor the hardware admits (a partial arrival hangs), so refuse.
            if not is_true(g):
                raise Exception('bar.sync under a predicate  (refusing, not guessing)')
            self.smem_snaps.append(self.smem)
            self.smem = self.bar.apply(self.smem, 'ptx')
        elif op == 'ld.global.v4.u32':
            dsts = re.fullmatch(r'\{(.*)\}', ops[0]).group(1).split(',')
            addr = self.D(re.fullmatch(r'\[(.*)\]', ops[1]).group(1))
            i = len(self.loads); self.loads.append((addr, g))
            for k, d in enumerate(dsts):
                v = (self.sym['abstract'](i, k) if 'abstract' in self.sym
                     else Select(self.mem, addr + BitVecVal(4*k, 64)))
                self.wr(d, v, g)
        elif op in ('ld.global.u32','ld.global.s32','ld.global.b32','ld.global.nc.u32'):
            addr = self.D(re.fullmatch(r'\[(.*)\]', ops[1]).group(1))
            i = len(self.loads); self.loads.append((addr, g))
            v = (self.sym['abstract'](i,0) if 'abstract' in self.sym
                 else Select(self.mem, addr))
            self.wr(ops[0], v, g)
        # --- a PTX macro-op: ptxas does not transliterate it, it inlines a
        # --- refinement sequence.  Named refusal, with the reason.
        elif op in fpmode.MACRO_OPS:
            fpmode.refuse_macro_op(op)
        # --- f32.  `mul.f32` and `mul.rn.f32` compute the SAME value; they
        # --- differ only in whether ptxas may contract them, which is a fact
        # --- about the compiler and not about the arithmetic.
        elif op in ('mul.f32','mul.rn.f32'):
            self.wf(ops[0], self.sym['fp']('FMUL', self.F(ops[1]), self.F(ops[2]), side='ptx'), g)
        elif op in ('add.f32','add.rn.f32'):
            self.wf(ops[0], self.sym['fp']('FADD', self.F(ops[1]), self.F(ops[2]), side='ptx'), g)
        elif op in ('sub.f32','sub.rn.f32'):
            self.wf(ops[0], self.sym['fp']('FSUB', self.F(ops[1]), self.F(ops[2]), side='ptx'), g)
        elif op in ('fma.rn.f32',):
            self.wf(ops[0], self.sym['fp']('FFMA', self.F(ops[1]), self.F(ops[2]), self.F(ops[3]), side='ptx'), g)
        elif op == 'mov.f32':
            self.wf(ops[0], self.F(ops[1]), g)
        elif op == 'ld.global.f32':
            addr = self.D(re.fullmatch(r'\[(.*)\]', ops[1]).group(1))
            i = len(self.loads); self.loads.append((addr, g))
            v = (self.sym['abstract'](i,0) if 'abstract' in self.sym else Select(self.mem, addr))
            self.wf(ops[0], v, g)
        # --- sub-word global loads.
        # MEMORY MODEL: `mem` is Array(BV64 -> BV32) -- the 32-bit word AT a byte
        # address.  A byte load takes the low byte of the word at that address.
        # That is NOT a faithful byte-addressed memory (a byte load at `a` and a
        # word load at `a` alias in it), but BOTH EXECUTORS USE EXACTLY THIS
        # CONVENTION, which is what equivalence needs; where the two programs
        # address differently the model reports a difference, which is the safe
        # direction.
        elif op in ('ld.global.s8','ld.global.u8','ld.global.s16','ld.global.u16'):
            addr = self.D(re.fullmatch(r'\[(.*)\]', ops[1]).group(1))
            i = len(self.loads); self.loads.append((addr, g))
            w = (self.sym['abstract'](i,0) if 'abstract' in self.sym
                 else Select(self.mem, addr))
            nb = 8 if op.endswith('8') else 16
            byte = Extract(nb-1, 0, w)
            self.wr(ops[0], (SignExt(W-nb, byte) if '.s' in op else ZeroExt(W-nb, byte)), g)
        elif op == 'st.global.f32':
            addr = self.SA(re.fullmatch(r'\[(.*)\]', ops[0]).group(1))
            self.stores.append((addr, self.F(ops[1]), g))
        elif op in ('st.global.u32','st.global.s32','st.global.b32'):
            addr = self.SA(re.fullmatch(r'\[(.*)\]', ops[0]).group(1))
            self.stores.append((addr, self.R(ops[1]), g))
        elif op == 'st.global.v4.u32':
            addr = self.SA(re.fullmatch(r'\[(.*)\]', ops[0]).group(1))
            srcs = re.fullmatch(r'\{(.*)\}', ops[1]).group(1).split(',')
            for k, s in enumerate(srcs):
                self.stores.append((addr + BitVecVal(4*k, 64), self.R(s), g))
        # --- the carry-flag family -------------------------------------
        elif op in ('mad.lo.cc.u32','madc.lo.cc.u32','mad.hi.cc.u32','madc.hi.cc.u32'):
            prod = (self.hi if '.hi.' in op else self.lo)(self.R(ops[1]), self.R(ops[2]))
            cin = self.cc if op.startswith('madc') else None
            s, co = self.addc(prod, self.R(ops[3]), cin)
            self.wr(ops[0], s, g); self.cc = If(g, co, self.cc); self.widen(s, co)
        elif op in ('add.cc.u32','addc.cc.u32','addc.u32','add.u32'):
            cin = self.cc if op.startswith('addc') else None
            s, co = self.addc(self.R(ops[1]), self.R(ops[2]), cin)
            self.wr(ops[0], s, g); self.widen(s, co)
            if op.endswith('.cc.u32'): self.cc = If(g, co, self.cc)
        elif op in ('sub.cc.u32','subc.cc.u32','subc.u32','sub.u32'):
            bin_ = self.cc if op.startswith('subc') else None
            s, co = self.subb(self.R(ops[1]), self.R(ops[2]), bin_)
            self.wr(ops[0], s, g); self.widen(s, co)
            if op.endswith('.cc.u32'): self.cc = If(g, co, self.cc)
        # --- address / move / convert ----------------------------------
        elif op in ('cvta.to.global.u64','cvta.global.u64'):
            self.wd(ops[0], self.D(ops[1]), g)          # address-space cast: identity here
        elif op in ('mov.u64','mov.s64','mov.b64'):
            self.wd(ops[0], self.D(ops[1]), g)
        elif op == 'cvt.u32.u64':   self.wr(ops[0], Extract(31,0,self.D(ops[1])), g)
        elif op in ('cvt.s64.s32',): self.wd(ops[0], SignExt(32, self.R(ops[1])), g)
        elif op == 'ret':           pass
        # --- integer ALU -----------------------------------------------
        elif op in ('sub.u32','sub.s32'): self.wr(ops[0], self.R(ops[1]) - self.R(ops[2]), g)
        elif op == 'neg.s32':       self.wr(ops[0], -self.R(ops[1]), g)
        elif op == 'shl.b32':       self.wr(ops[0], self.R(ops[1]) << self.R(ops[2]), g)
        elif op in ('shr.u32','shr.b32'): self.wr(ops[0], LShR(self.R(ops[1]), self.R(ops[2])), g)
        elif op == 'shr.s32':       self.wr(ops[0], self.R(ops[1]) >> self.R(ops[2]), g)
        elif op in ('mad.lo.u32','mad.lo.s32'):
            self.wr(ops[0], self.lo(self.R(ops[1]), self.R(ops[2])) + self.R(ops[3]), g)
        elif op == 'mul.wide.u32':
            self.wd(ops[0], ZeroExt(32,self.R(ops[1])) * ZeroExt(32,self.R(ops[2])), g)
        elif op == 'mul.wide.s32':
            self.wd(ops[0], SignExt(32,self.R(ops[1])) * SignExt(32,self.R(ops[2])), g)
        elif op == 'div.u32':       self.wr(ops[0], UDiv(self.R(ops[1]), self.R(ops[2])), g)
        elif op == 'div.s32':       self.wr(ops[0], self.R(ops[1]) / self.R(ops[2]), g)
        elif op == 'rem.u32':       self.wr(ops[0], URem(self.R(ops[1]), self.R(ops[2])), g)
        elif op == 'rem.s32':       self.wr(ops[0], SRem(self.R(ops[1]), self.R(ops[2])), g)
        elif op in ('min.u32',):    self.wr(ops[0], If(ULE(self.R(ops[1]),self.R(ops[2])), self.R(ops[1]), self.R(ops[2])), g)
        elif op in ('max.u32',):    self.wr(ops[0], If(UGE(self.R(ops[1]),self.R(ops[2])), self.R(ops[1]), self.R(ops[2])), g)
        elif op in ('min.s32',):    self.wr(ops[0], If(self.R(ops[1])<=self.R(ops[2]), self.R(ops[1]), self.R(ops[2])), g)
        elif op in ('max.s32',):    self.wr(ops[0], If(self.R(ops[1])>=self.R(ops[2]), self.R(ops[1]), self.R(ops[2])), g)
        elif op == 'selp.u32':      self.wr(ops[0], If(self.P(ops[3]), self.R(ops[1]), self.R(ops[2])), g)
        elif op in ('mul.lo.u64','mul.lo.s64'): self.wd(ops[0], self.D(ops[1])*self.D(ops[2]), g)
        elif op == 'sub.u64':       self.wd(ops[0], self.D(ops[1]) - self.D(ops[2]), g)
        elif op in ('add.s64','add.u64.'): self.wd(ops[0], self.D(ops[1]) + self.D(ops[2]), g)
        elif op == 'shr.u64':       self.wd(ops[0], LShR(self.D(ops[1]), ZeroExt(32,self.R(ops[2])) if re.fullmatch(r'%r\d+',ops[2].strip()) else BitVecVal(int(ops[2]),64)), g)
        # --- comparisons ------------------------------------------------
        elif op.startswith('setp.'):
            k = op.split('.')[1]; ty = op.split('.')[2] if len(op.split('.'))>2 else 'u32'
            wide = ty in ('u64','s64','b64')
            a = self.D(ops[1]) if wide else self.R(ops[1])
            b = self.D(ops[2]) if wide else self.R(ops[2])
            sgn = ty.startswith('s')
            f = {'lt': (lambda x,y: x<y) if sgn else ULT, 'le': (lambda x,y: x<=y) if sgn else ULE,
                 'gt': (lambda x,y: x>y) if sgn else UGT, 'ge': (lambda x,y: x>=y) if sgn else UGE,
                 'eq': lambda x,y: x==y, 'ne': lambda x,y: x!=y}.get(k)
            if f is None: raise Exception(f'UNMODELLED PTX COMPARISON {op!r}')
            self.wp(ops[0], f(a,b), g)
        # --- wider global access ---------------------------------------
        elif op in ('ld.global.v2.u32','ld.global.v2.b32'):
            dsts = re.fullmatch(r'\{(.*)\}', ops[0]).group(1).split(',')
            addr = self.D(re.fullmatch(r'\[(.*)\]', ops[1]).group(1))
            i = len(self.loads); self.loads.append((addr, g))
            for k2,d in enumerate(dsts):
                v = (self.sym['abstract'](i,k2) if 'abstract' in self.sym
                     else Select(self.mem, addr + BitVecVal(4*k2,64)))
                self.wr(d, v, g)
        elif op in ('st.global.v2.u32','st.global.v2.b32'):
            addr = self.SA(re.fullmatch(r'\[(.*)\]', ops[0]).group(1))
            srcs = re.fullmatch(r'\{(.*)\}', ops[1]).group(1).split(',')
            for k2,s2 in enumerate(srcs):
                self.stores.append((addr + BitVecVal(4*k2,64), self.R(s2), g))
        elif op == 'st.global.u64':
            addr = self.SA(re.fullmatch(r'\[(.*)\]', ops[0]).group(1))
            v = self.D(ops[1])
            self.stores.append((addr, Extract(31,0,v), g))
            self.stores.append((addr+BitVecVal(4,64), Extract(63,32,v), g))
        else:
            raise Exception(f'UNMODELLED PTX OPCODE {op!r}  (refusing, not guessing)')

def run_lines(lines, sym, seed=None):
    """Run an explicit list of instruction strings.

    Used to execute ONE REGION of a kernel -- a prologue, a loop body, an
    epilogue -- in a fresh state.  A register read before it is written then
    materialises as `ptx_undef_rN`, which is exactly the region's live-in, and
    the naming is deterministic so the same live-in has the same name in every
    run of that region."""
    st = Ptx(sym)
    # See sassexec.run_insns for why a region may need to be run in another
    # vocabulary rather than rewritten afterwards.
    if seed:
        for (kind, i), v in seed.items():
            getattr(st, kind)[i] = v
    for s in lines:
        st.step(s)
    return st


def run_ptx(path, sym):
    sym.setdefault('smem_layout', {}).update(smem.layout(path))
    st = Ptx(sym); started = False
    for line in open(path):
        s = line.strip()
        if s.startswith('//') or not s: continue
        if s.startswith('.') or s.startswith('.reg') or s.endswith('(') or s.startswith('.param'): continue
        if s == '{': started = True; continue
        if s == '}': break
        if not started: continue
        if not s.endswith(';'): continue
        st.step(s[:-1].strip())
    return st

if __name__ == '__main__':
    sym = {k: BitVec(k, W) for k in
           ('A_lo','A_hi','B_lo','B_hi','O_lo','O_hi','N','stackptr',
            'gridc_lo','gridc_hi','ctaid_x','tid_x')}
    sym['mem'] = Array('mem', BitVecSort(64), BitVecSort(32))
    st = run_ptx(sys.argv[1], sym)
    print(f'executed {st.count} instructions, {len(st.ops)} opcodes')
    print(f'stores: {len(st.stores)}   undefined reads: {st.undef}')
