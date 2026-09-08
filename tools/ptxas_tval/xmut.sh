#!/bin/bash
# Mutation table for the MAX work: `max.f32`/`min.f32`, the FMNMX polarity
# operand, the FMAX commutativity identification, and the WIDENED gate.
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
# THE `max/` FIXTURES ARE NEW AND UNTRACKED WHEN THIS FIRST RUNS.  `mkbase.sh`
# derives its member list from `git ls-files -co --exclude-standard` for exactly
# that reason: archiving only TRACKED files would leave a probe that rewrites
# max/*.sass restored by nothing, which is the failure the coverage check exists
# to prevent, reintroduced by the fix for it.
cd "$(dirname "$0")"
./mkbase.sh max_base.tgz || exit 1

run(){   # $1 = label
  if python3 -c 'import fpmode' >/dev/null 2>&1; then imp=ok; else imp=FAIL; fi
  timeout 900 ./regress.sh >/tmp/_xr.txt 2>&1; r=$?
  n=$(grep -c 'VALIDATED\|UNPROVED' /tmp/_xr.txt)
  timeout 300 python3 fpsem_abi.py >/tmp/_xs.txt 2>&1; f=$?
  timeout 900 python3 fpgate.py   >/tmp/_xg.txt 2>&1; g=$?
  # WHICH of the three max rows moved, not just that regress failed: a row that
  # says only FAIL is a detection, and the diagnosis is the point of the table.
  v(){ case "$(grep "^$1 " /tmp/_xr.txt | awk '{print $2}')" in
         VALIDATED) echo V;; UNPROVED) echo U;; *) echo '?';; esac; }
  printf '%-60s  import=%-4s regress=%-4s(%2s rows, relu/general/min=%s%s%s) fpsem=%-4s fpgate=%s\n' \
         "$1" "$imp" "$([ $r = 0 ] && echo ok || echo FAIL)" "$n" \
         "$(v relu)" "$(v general)" "$(v min)" \
         "$([ $f = 0 ] && echo ok || echo FAIL)" "$([ $g = 0 ] && echo ok || echo FAIL)"
}

M(){ ./restore.sh >/dev/null; python3 - ; rm -rf __pycache__; }   # stdin = patch

printf '%-60s  %s\n' 'MUTATION' 'four checks, each run separately'
./restore.sh >/dev/null; rm -rf __pycache__
run 'X0 CONTROL: two MAX_PAIRS entries reordered (no-op)'
M <<'EOF'
s=open('fpsem_abi.py').read()
a="    (0x7F800000, 0x7FC0DEAD), (0xFF800000, 0x00000000),   # infinities against NaN and zero\n    (0x7F800000, 0xFF800000), (0x3F800000, 0xBF800000),   # +inf vs -inf; 1 vs -1\n"
b="    (0x7F800000, 0xFF800000), (0x3F800000, 0xBF800000),   # +inf vs -inf; 1 vs -1\n    (0x7F800000, 0x7FC0DEAD), (0xFF800000, 0x00000000),   # infinities against NaN and zero\n"
assert s.count(a)==1; open('fpsem_abi.py','w').write(s.replace(a,b))
EOF
run 'X0 CONTROL (applied)'

M <<'EOF'
s=open('ptxexec.py').read()
a="        elif op in ('max.f32','min.f32'):"
assert s.count(a)==1; open('ptxexec.py','w').write(s.replace(a,"        elif op in ('__gone.f32',):"))
EOF
run 'X1 max.f32/min.f32 unmodelled again (the state before this change)'

M <<'EOF'
s=open('sassexec.py').read()
a="        elif opc == 'FMNMX':"
assert s.count(a)==1; open('sassexec.py','w').write(s.replace(a,"        elif opc == '__GONE':"))
EOF
run 'X2 FMNMX unmodelled on the SASS side'

M <<'EOF'
s=open('sassexec.py').read()
a="""            self.wr(ops[0], If(self.pr(ops[3]),
                               self.sym['fp']('FMIN', self.frd(ops[1]), self.frd(ops[2]), side='sass'),
                               self.sym['fp']('FMAX', self.frd(ops[1]), self.frd(ops[2]), side='sass')), g)"""
b="""            self.wr(ops[0], If(self.pr(ops[3]),
                               self.sym['fp']('FMAX', self.frd(ops[1]), self.frd(ops[2]), side='sass'),
                               self.sym['fp']('FMIN', self.frd(ops[1]), self.frd(ops[2]), side='sass')), g)"""
assert s.count(a)==1; open('sassexec.py','w').write(s.replace(a,b))
EOF
run 'X3 FMNMX polarity INVERTED (min and max swapped)'

M <<'EOF'
s=open('fpmode.py').read()
a="    'FMAX_IS_COMMUTATIVE': True,"
assert s.count(a)==1; open('fpmode.py','w').write(s.replace(a,"    'FMAX_IS_COMMUTATIVE': False,"))
EOF
run 'X4 the commutativity identification marked unvalidated'

M <<'EOF'
s=open('fpmode.py').read()
a="        if name == 'FMAX' and len(args) == 2:"
assert s.count(a)==1; open('fpmode.py','w').write(s.replace(a,"        if name in ('FMAX','FMIN') and len(args) == 2:"))
EOF
run 'X5 FMIN canonicalised too (an unmeasured identification)'

M <<'EOF'
s=open('fpmode.py').read()
a="          for n in ('FMUL', 'FADD', 'FSUB', 'FMAX', 'FMIN')}"
assert s.count(a)==1
s=s.replace(a,"          for n in ('FMUL', 'FADD', 'FSUB', 'FMAX')}\n    F2['FMIN'] = F2['FMAX']")
open('fpmode.py','w').write(s)
EOF
run 'X6 FMAX and FMIN collapsed onto one symbol'

M <<'EOF'
s=open('fpgate.py').read()
a="    return (None, None)"
assert s.count(a)==1
open('fpgate.py','w').write(s.replace(a,"    return ('unclassified', 'no reason given')"))
EOF
run 'X7 why_unmodelled gives every opcode a family'

M <<'EOF'
s=open('fpgate.py').read()
a="SEM_MNEM = frozenset(('mul', 'add', 'sub', 'neg', 'fma', 'mad', 'div', 'rcp',"
assert s.count(a)==1
open('fpgate.py','w').write(s.replace(a,"SEM_MNEM = frozenset(('__none',))\n_UNUSED = (("))
EOF
run 'X8 the float-semantic predicate matches nothing (floor)'

M <<'EOF'
s=open('fpgate.py').read()
a="""        except Exception:"""
b="""        except Exception as _e:
            if 'UNMODELLED PTX OPCODE' not in str(_e) and 'MACRO-OP' not in str(_e):
                modelled.append(op); continue"""
assert s.count(a)==1; open('fpgate.py','w').write(s.replace(a,b,1))
EOF
run 'X9 the operand-refusal exemption restored (setp.lt.f64 reads as modelled)'

M <<'EOF'
s=open('fpsem_abi.py').read()
a="(0xBF800000, 0x00000000),"
assert s.count(a)==1
open('fpsem_abi.py','w').write(s.replace(a,"(0xBF800000, 0xBF800000),"))
EOF
run 'X10 a MAX_PAIRS entry made self-identical (commutativity vacuous)'

M <<'EOF'
s=open('regress.sh').read()
a="    plain|unfoldable) echo \"$out\" | grep -q '^UNPROVED'  || bad=$((bad+1)) ;;"
b="    plain|unfoldable|relu) echo \"$out\" | grep -q '^UNPROVED'  || bad=$((bad+1)) ;;"
assert s.count(a)==1; open('regress.sh','w').write(s.replace(a,b))
EOF
run 'X11 max/relu asserted UNPROVED instead of VALIDATED'

./restore.sh >/dev/null; rm -rf __pycache__
run 'RESTORED BASELINE (read this second)'
# ...AND `guard_base.tgz`, which is the one `restore.sh` actually extracts.
# Each harness used to delete only the archive it created, so a run that
# finished normally still left behind the archive that RESTORES -- and the
# next run, after a source had been edited, silently put the old tree back.
# That is not hypothetical: it reverted this file's own X10 fix mid-session.
# mkbase's stderr notice is not a guard, because it prints on every restore
# of a run, so a stale one looks like a fresh one apart from a timestamp.
rm -f max_base.tgz guard_base.tgz
