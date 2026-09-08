#!/bin/bash
# Mutation table for the NEGATION work: `neg.f32`, the `-R` float operand
# modifier, and the FSUB == FADD-of-FNEG identification.
#
# CONTROL ROW IS FIRST and must be read first: a table where every row moves the
# same column is reporting the state of the tree, not the mutations.  The
# RESTORED BASELINE row at the bottom is read second -- if it does not match the
# control, the archive did not restore and every row above it is contaminated.
#
# Four checks, each run separately.  `import` is its own column because
# `fpmode` self-checks AT IMPORT, so a mutation of the abstraction fails
# everything at once and that is a different diagnosis from a kernel moving.
#
# G9 MUTATES `regress.sh`, WHICH IS NOT A SOURCE FILE.  The first run of this
# table had a RED restored-baseline row for exactly that reason: `mkbase.sh`
# archived `*.py` and the fixtures, so the restore put every source back and
# left `regress.sh` carrying G9's mutation.  Rows G0..G9 were unaffected --
# each restores BEFORE applying its own mutation, and G9 is the only probe that
# writes outside the archive -- but the baseline row is the row that says so,
# and it said so.  `mkbase.sh` now archives every `.py` and `.sh` except the
# two that extract it, and CHECKS that coverage against `git ls-files`.
cd "$(dirname "$0")"
./mkbase.sh neg_base.tgz || exit 1

run(){   # $1 = label
  if python3 -c 'import fpmode' >/dev/null 2>&1; then imp=ok; else imp=FAIL; fi
  timeout 900 ./regress.sh >/tmp/_nr.txt 2>&1; r=$?
  n=$(grep -c 'VALIDATED\|UNPROVED' /tmp/_nr.txt)
  timeout 300 python3 fpsem_abi.py >/tmp/_ns.txt 2>&1; f=$?
  timeout 900 python3 fpgate.py   >/tmp/_ng.txt 2>&1; g=$?
  # WHICH of the three neg rows moved, not just that regress failed: a row that
  # says only FAIL is a detection, and the diagnosis is the point of the table.
  v(){ case "$(grep "^$1 " /tmp/_nr.txt | awk '{print $2}')" in
         VALIDATED) echo V;; UNPROVED) echo U;; *) echo '?';; esac; }
  printf '%-58s  import=%-4s regress=%-4s(%2s rows, folded/sub/unfold=%s%s%s) fpsem=%-4s fpgate=%s\n' \
         "$1" "$imp" "$([ $r = 0 ] && echo ok || echo FAIL)" "$n" \
         "$(v folded)" "$(v sub)" "$(v unfoldable)" \
         "$([ $f = 0 ] && echo ok || echo FAIL)" "$([ $g = 0 ] && echo ok || echo FAIL)"
}

M(){ ./restore.sh >/dev/null; python3 - ; rm -rf __pycache__; }   # stdin = patch

printf '%-58s  %s\n' 'MUTATION' 'four checks, each run separately'
./restore.sh >/dev/null; rm -rf __pycache__; run 'G0 CONTROL: two independent self-checks reordered (no-op)'

M <<'P'
s=open('fpmode.py').read()
a="""    if is_true(simplify(f('FNEG', a, side='sass') == a)):"""
b="""    if is_true(simplify(f('FNEG', a, side='sass') == f('FNEG', b, side='sass'))):"""
assert s.count(a)==1 and s.count(b)==1
i,j=s.index(a),s.index(b)
blk1=s[i:j]; k=s.index("    if not is_true(simplify(f('FSUB'",j); blk2=s[j:k]
open('fpmode.py','w').write(s[:i]+blk2+blk1+s[k:])
P
run 'G0 CONTROL (applied)'

# --- G1: the ORIGINAL STATE of the PTX half -- `neg.f32` unmodelled, which is
# --- what the RoPE fma repair handed the validator without anyone measuring it.
M <<'P'
s=open('ptxexec.py').read()
i=s.index("        elif op in ('neg.f32',):"); j=s.index("        elif op == 'mov.f32':")
open('ptxexec.py','w').write(s[:i]+s[j:])
P
run 'G1 neg.f32 unmodelled again (the state the fma repair left)'

# --- G2: the ORIGINAL STATE of the SASS half -- a float source read by the
# --- INTEGER reader, so `-R` is a two's complement of the bit pattern.
M <<'P'
s=open('sassexec.py').read()
s=s.replace("self.sym['fp'](opc, *[self.frd(o) for o in ops[1:1+n]], side='sass')",
            "self.sym['fp'](opc, *[rd(o) for o in ops[1:1+n]], side='sass')",1)
open('sassexec.py','w').write(s)
P
run 'G2 float sources read by the integer reader (the guess)'

# --- G3: FNEG is the identity.  Then `FADD Ra, -Rb` and `FADD Ra, Rb` are the
# --- same term and a kernel that negates the WRONG source validates.
M <<'P'
s=open('fpmode.py').read()
s=s.replace("            return f('FADD', args[0], f('FNEG', args[1], side=side), side=side)",
            "            return f('FADD', args[0], args[1], side=side)",1)
s=s.replace("        t = {1: F1, 2: F2, 3: F3}[len(args)]",
            "        if name == 'FNEG': return args[0]\n        t = {1: F1, 2: F2, 3: F3}[len(args)]",1)
open('fpmode.py','w').write(s)
P
run 'G3 FNEG is the identity'

# --- G4: the FSUB rewrite deleted.  FSUB is not a symbol, so this is a HARD
# --- ERROR at the first `sub.f32` rather than a silent revert to two terms.
M <<'P'
s=open('fpmode.py').read()
i=s.index("        if name == 'FSUB':"); j=s.index("        if name == 'FADD' and len(args) == 2")
open('fpmode.py','w').write(s[:i]+s[j:])
P
run 'G4 the FSUB == FADD(a,FNEG(b)) rewrite deleted'

# --- G5: the identification imposed with its validated flag OFF.
M <<'P'
s=open('fpmode.py').read()
s=s.replace("""            if not IDENTIFICATIONS['FSUB_IS_FADD_OF_FNEG']:
                raise Exception('FSUB is identified with FADD(a, FNEG(b)) but the '
                                'device probe that settles it is marked unvalidated')
""","",1)
open('fpmode.py','w').write(s)
P
run 'G5 the identification no longer reads its validated flag'

# --- G6: the un-foldable referee compares the device against ITSELF, so nothing
# --- can differ.  Its non-vacuity assertion is the whole point of that probe.
M <<'P'
s=open('fpsem_abi.py').read()
s=s.replace("        flip = v ^ 0x80000000\n        if o == flip:","        flip = o\n        if o == flip:",1)
open('fpsem_abi.py','w').write(s)
P
run 'G6 the neg refutation compares the device against itself'

# --- G7: the subtract referee uses a LITERAL sign mask.  ptxas then recognises
# --- the xor as a negation and CSEs both arms into ONE instruction.
M <<'P'
s=open('fpsem_abi.py').read()
s=s.replace("    ld.global.u32 %r4, [%rd22];\n    sub.f32","    mov.u32 %r4, -2147483648;\n    sub.f32",1)
open('fpsem_abi.py','w').write(s)
P
run 'G7 the subtract referee uses a literal mask (ptxas CSEs the arms)'

# --- G8: the gap-growth gate scans nothing.  Its FLOOR must fire; without one,
# --- a scan that reads no artifact reports no unmodelled opcodes perfectly.
M <<'P'
s=open('fpgate.py').read()
s=s.replace("    files = sorted(glob.glob(os.path.join('..', '..', 'tests', '*.ptx')))",
            "    files = sorted(glob.glob(os.path.join('..', '..', 'tests', '*.nope')))",1)
open('fpgate.py','w').write(s)
P
run 'G8 the gap-growth gate scans no artifacts'

# --- G9: `neg/unfoldable` asserted as VALIDATED.  It is a standing REFUTATION
# --- and a run in which it turns green is a regression.
M <<'P'
s=open('regress.sh').read()
s=s.replace("    plain|unfoldable) echo \"$out\" | grep -q '^UNPROVED'","    plain) echo \"$out\" | grep -q '^UNPROVED'",1)
open('regress.sh','w').write(s)
P
run 'G9 neg/unfoldable asserted VALIDATED instead of UNPROVED'

./restore.sh >/dev/null; rm -rf __pycache__; run 'RESTORED BASELINE (read this second)'
# ...AND `guard_base.tgz`, which is the one `restore.sh` actually extracts.
# Each harness used to delete only the archive it created, so a run that
# finished normally still left behind the archive that RESTORES -- and the
# next run, after a source had been edited, silently put the old tree back.
# That is not hypothetical: it reverted this file's own X10 fix mid-session.
# mkbase's stderr notice is not a guard, because it prints on every restore
# of a run, so a stale one looks like a fresh one apart from a timestamp.
rm -f neg_base.tgz guard_base.tgz
