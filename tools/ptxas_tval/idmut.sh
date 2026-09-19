#!/bin/bash
# Mutation table for the u32 division lowering in the validator: divest.py (the
# estimate as a fresh value with the device facts), intenc.py (the exact
# bitvector-to-Int rung), the unspecified-at-zero model in ptxexec.py and its
# obligations in tval.py.  CONTROL ROW FIRST, BASE at the top and the bottom.
#
# One verdict LETTER per fixture, in this order:
#   ptx_integer_ops no_second_corr rem_wrong rem_any_at_d0 bias_moved twice twice_split and1 and1_two
# BASE reads  V U U V R V U U U.  `imp` is whether every module imports (they
# self-check at import, so a mutation of an abstraction fails there by design).
#
# Every step names its target LITERALLY in open('...') so mutgate.py can see it.
cd "$(dirname "$0")"
./mkbase.sh guard_base.tgz || exit 1
FIX="corpus/ptx_integer_ops idiv/no_second_corr idiv/rem_wrong idiv/rem_any_at_d0 idiv/bias_moved udiv/twice udiv/twice_split udiv/and1 udiv/and1_two"
row(){   # $1 = label
  touch *.py
  if python3 -c "import tval, intenc, divest, sassexec, ptxexec" >/dev/null 2>&1; then imp=ok; else imp=FAIL; fi
  v=""
  for p in $FIX; do
    o=$(timeout 600 python3 tval.py "$p.ptx" "$p.sass" 12 3 15 2>&1 | tail -1)
    case "$o" in VALIDATED*) v="${v}V";; UNPROVED*) v="${v}U";; REFUSED*) v="${v}R";; *) v="${v}C";; esac
  done
  printf '%-62s imp=%-4s %s\n' "$1" "$imp" "$v"
}
printf '%-62s %-8s %s\n' MUTATION IMPORT 'int nsc rw d0 bias tw tws a1 a1t'
./restore.sh >/dev/null; row "BASE"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tval.py').read(); a="    proved=[]; first={}\n"
assert s.count(a)==1; open('tval.py','w').write(s.replace(a,"    first={}; proved=[]\n"))
P
row "C0 CONTROL: two independent initialisations reordered"

./restore.sh >/dev/null
python3 - <<'P'
s=open('sassexec.py').read(); a="        if self.est_link(opc, ops, g):\n            return\n"
assert s.count(a)==1; open('sassexec.py','w').write(s.replace(a,""))
P
row "M1 the estimate chain unmodelled (the ORIGINAL state)"

# M2 (the facts recorded on their own spelling removed) is RETIRED: its first
# run left every row unchanged, because only the TRANSFERRED Lemma A is ever used
# by a proof -- so the unused facts were deleted, and the row with them.

./restore.sh >/dev/null
python3 - <<'P'
s=open('sassexec.py').read(); a="                self.est_transfer(a1, ops[0], g)\n"
assert s.count(a)==1; open('sassexec.py','w').write(s.replace(a,""))
P
row "M3 Lemma A not transferred onto the SASS Newton node"

./restore.sh >/dev/null
python3 - <<'P'
s=open('divest.py').read(); a="And(ULE(I64 - BitVecVal(1, 64, C), x), ULE(x, I64))"
assert s.count(a)==1; open('divest.py','w').write(s.replace(a,"And(ULE(I64 - BitVecVal(2, 64, C), x), ULE(x, I64))"))
s=open('divest.py').read(); a="_self_check()\n"
assert s.endswith(a); open('divest.py','w').write(s[:-len(a)])
P
row "M4 Lemma A one unit wider (I-2), self-check off"

./restore.sh >/dev/null
python3 - <<'P'
s=open('intenc.py').read(); a="            a, b = [self.unreduced(self.bv(c), w) for c in ch]\n            return self.wrap(a - b, w, lower_ok=False)"
assert s.count(a)==1; open('intenc.py','w').write(s.replace(a,"            a, b = [self.unreduced(self.bv(c), w) for c in ch]\n            return self.wrap(b - a, w, lower_ok=False)"))
P
row "M5 intenc: subtraction reversed"

./restore.sh >/dev/null
python3 - <<'P'
s=open('intenc.py').read(); a="            a, b = [self.unreduced(self.bv(c), w) for c in ch]\n            return self.wrap(a - b, w, lower_ok=False)"
assert s.count(a)==1; s=s.replace(a,"            a, b = [self.unreduced(self.bv(c), w) for c in ch]\n            return self.wrap(b - a, w, lower_ok=False)")
a="\n_self_check()\n"
assert s.endswith(a); open('intenc.py','w').write(s[:-len(a)]+"\n")
P
row "M5b M5 with the encoder self-check off (the compound)"

./restore.sh >/dev/null
python3 - <<'P'
s=open('intenc.py').read(); a="        return p[0] if p is not None and p[1] == w else a"
assert s.count(a)==1; open('intenc.py','w').write(s.replace(a,"        return a"))
P
row "M6 intenc: deferred reduction off"

./restore.sh >/dev/null
python3 - <<'P'
s=open('intenc.py').read(); a="            return self.I(v - (1 << w)) if v >= (1 << (w - 1)) else a"
assert s.count(a)==1; open('intenc.py','w').write(s.replace(a,"            return a"))
P
row "M7 intenc: constants keep their unsigned representative"

./restore.sh >/dev/null
python3 - <<'P'
s=open('intenc.py').read(); a="            key = ('divq', a.get_id(), b.get_id(), w)"
assert s.count(a)==1; open('intenc.py','w').write(s.replace(a,"            key = ('divq', a.get_id(), b.get_id(), w, t.get_id())"))
P
row "M8 intenc: UDiv and URem get separate quotients"

./restore.sh >/dev/null
python3 - <<'P'
s=open('ptxexec.py').read(); a="            self.wr(ops[0], self.unspecified(b == BitVecVal(0, 32), f(a, b)), g)"
assert s.count(a)==1; open('ptxexec.py','w').write(s.replace(a,"            self.wr(ops[0], f(a, b), g)"))
P
row "M9 PTX division by zero back to z3's convention"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tval.py').read(); a="            if u.get_id() in first:\n"
assert s.count(a)==1; open('tval.py','w').write(s.replace(a,"            if False:\n"))
P
row "M10 the one-unspecified-value consistency check removed"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tval.py').read(); a="            if str(sv.check()) != 'unsat':\n                bad = f'store {k}: stores a value computed FROM"
assert s.count(a)==1; open('tval.py','w').write(s.replace(a,"            if False:\n                bad = f'store {k}: stores a value computed FROM"))
P
row "M11 the stores-the-value-itself check removed"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tval.py').read(); a="        return [f for f in facts if ests(f, set(), set()) & mine]"
assert s.count(a)==1; open('tval.py','w').write(s.replace(a,"        return []"))
P
row "M12 no fact is ever relevant"

./restore.sh >/dev/null
python3 - <<'P'
s=open('tval.py').read(); a="            proved.append(Implies(And(extra), sd == pd) if extra else sd == pd)\n"
assert s.count(a)==1; open('tval.py','w').write(s.replace(a,"            pass\n"))
P
row "M13 a proved store is not a fact for later stores"

./restore.sh >/dev/null
python3 - <<'P'
s=open('sassexec.py').read(); a="            if opc == 'IADD3' and (len(ops) != 4 or ops[2] != hex(divest.BIAS)"
assert s.count(a)==1; open('sassexec.py','w').write(s.replace(a,"            if False and (len(ops) != 4 or ops[2] != hex(divest.BIAS)"))
P
row "M14 the estimate's bias not checked"

./restore.sh >/dev/null
python3 - <<'P'
s=open('sassexec.py').read(); a="            lo = Extract(W-1, 0, Concat(rd(ops[3]), rd(ops[1])) << ZeroExt(32, n))"
assert s.count(a)==1; open('sassexec.py','w').write(s.replace(a,"            lo = Extract(W-1, 0, LShR(Concat(rd(ops[3]), rd(ops[1])), ZeroExt(32, n)))"))
P
row "M15 SHF.L.U32 modelled as a right shift"

./restore.sh >/dev/null
python3 - <<'P'
s=open('sassexec.py').read(); a="            if isinstance(self.R[i], divest.Tagged):\n"
assert s.count(a)==1; open('sassexec.py','w').write(s.replace(a,"            if False:\n"))
P
row "M16 a tagged intermediate readable outside the chain"

./restore.sh >/dev/null
echo; echo "restored; verifying baseline is back:"; row "BASE"

rm -f guard_base.tgz  # a surviving archive means this run did not finish
