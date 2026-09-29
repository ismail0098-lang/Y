#!/bin/bash
# Rebuild corpus/ from the repository's own committed PTX.
#
#   tests/<k>.ptx  --ptxas-->  <k>.cubin  --nvdisasm-->  corpus/<k>.sass
#
# corpus/<k>.ptx is a COPY of tests/<k>.ptx, byte for byte.  The validator's
# whole question is whether ptxas preserved the meaning of a file this repo
# already ships, so the input must be that file and not a re-emission of it.
#
# Nothing here is committed.  A .cubin is a machine-specific ELF and a .sass is
# a disassembly of one; both are derived, both are large, and this repo already
# refuses to commit binaries for the same reason (see .gitignore).  Run this
# once and the corpus is there.
#
# THE ARCH COMES FROM EACH FILE'S OWN `.target`, never from the local card.
# Compiling at the build machine's architecture is exactly the bug
# `tests/ptx_portability.rs` exists to prevent, and it would silently change
# which SASS is under test.
set -u
cd "$(dirname "$0")"
REPO=../..
OUT=corpus
# CLEAN, do not merely overwrite.  A source deleted from tests/ otherwise leaves
# its artifact behind in the corpus, so the validator keeps measuring a kernel
# the repository no longer ships -- the stale-artifact class this whole corpus
# exists downstream of, one layer down.  Found exactly that way, when
# hello.coprocessor.ptx was deleted and the corpus still reported 67.
rm -rf "$OUT" o1 idiv
mkdir -p "$OUT"
command -v ptxas    >/dev/null || { echo "ptxas not on PATH (CUDA toolkit)";    exit 1; }
command -v nvdisasm >/dev/null || { echo "nvdisasm not on PATH (CUDA toolkit)"; exit 1; }

ok=0; skip=0
for f in "$REPO"/tests/*.ptx; do
  k=$(basename "$f" .ptx)
  arch=$(grep -m1 -oE '^\.target[[:space:]]+sm_[0-9]+[a-z]*' "$f" | awk '{print $2}')
  [ -n "$arch" ] || { echo "  skip $k: no .target"; skip=$((skip+1)); continue; }
  if ! ptxas -arch="$arch" -o "$OUT/$k.cubin" "$f" 2>"$OUT/$k.err"; then
    echo "  skip $k: ptxas: $(head -1 "$OUT/$k.err" | cut -c1-70)"
    rm -f "$OUT/$k.cubin" "$OUT/$k.err"; skip=$((skip+1)); continue
  fi
  rm -f "$OUT/$k.err"
  if ! nvdisasm -c "$OUT/$k.cubin" > "$OUT/$k.sass" 2>/dev/null; then
    echo "  skip $k: nvdisasm failed"; rm -f "$OUT/$k.sass"; skip=$((skip+1)); continue
  fi
  cp "$f" "$OUT/$k.ptx"
  ok=$((ok+1))
done

# The loop validator's subject is the SAME kernel at a LOWER ptxas level: it
# validates at -O1 and not at -O2/-O3, where ptxas unrolls the loop x4.  That
# differential is the result, so the -O1 build is part of the corpus.
mkdir -p o1
if [ -f "$REPO/tests/exact_pv.ptx" ]; then
  arch=$(grep -m1 -oE '^\.target[[:space:]]+sm_[0-9]+[a-z]*' "$REPO/tests/exact_pv.ptx" | awk '{print $2}')
  ptxas -O1 -arch="$arch" -o o1/exact_pv.cubin "$REPO/tests/exact_pv.ptx" 2>/dev/null \
    && nvdisasm -c o1/exact_pv.cubin > o1/exact_pv.sass \
    && cp "$REPO/tests/exact_pv.ptx" o1/exact_pv.ptx \
    && echo "  o1/exact_pv rebuilt at -O1"
fi

# y_cpu_matmul is the NESTED subject (nestval.py): three loops, a store in the
# middle one, rotated at -O1 with each child's zero-trip guard in its parent's
# body and an EXIT guarding the whole nest.  Its five wrong twins are derived
# here from the genuine output, each by ONE asserted instruction substitution,
# so a ptxas that changes that output fails this build instead of silently
# testing a different program:
#   w1_store_stride    the store's row stride is K, not N          -> iteration store address
#   w2_acc_add         the accumulator adds instead of multiply-adds -> iteration store value
#   w3_acc_init        the inner accumulator starts at 1, not 0      -> never proposed
#   w7_acc_init_k0     the inner accumulator is 1.0 exactly when K == 0 -> BASE (child):
#                      random simulation cannot draw K == 0, so the pair IS
#                      proposed, and the loop is skipped and that value stored
#   w4_top_guard       the EXIT guard tests N, not M                 -> ENTRY (top level)
#   w5_inner_guard     the inner zero-trip guard tests N, not K      -> ENTRY (child)
#   w6_inner_backedge  the inner back edge tests N, not K            -> LOOPCOND
if [ -f "$REPO/tests/y_cpu_matmul.ptx" ]; then
  arch=$(grep -m1 -oE '^\.target[[:space:]]+sm_[0-9]+[a-z]*' "$REPO/tests/y_cpu_matmul.ptx" | awk '{print $2}')
  ptxas -O1 -arch="$arch" -o o1/y_cpu_matmul.cubin "$REPO/tests/y_cpu_matmul.ptx" 2>/dev/null \
    && nvdisasm -c o1/y_cpu_matmul.cubin > o1/y_cpu_matmul.sass \
    && cp "$REPO/tests/y_cpu_matmul.ptx" o1/y_cpu_matmul.ptx \
    && echo "  o1/y_cpu_matmul rebuilt at -O1" || exit 1
  python3 - o1 <<'PY' || exit 1
import sys
d = sys.argv[1]
src = open(f'{d}/y_cpu_matmul.sass').read()
ptx = open(f'{d}/y_cpu_matmul.ptx').read()
I = '                   '
twins = {
 'w1_store_stride':   ('/*01f0*/' + I + 'IMAD R3, R0.reuse, c[0x0][0x17c], R7 ;',
                       '/*01f0*/' + I + 'IMAD R3, R0.reuse, c[0x0][0x180], R7 ;'),
 'w2_acc_add':        ('/*01c0*/' + I + 'FFMA R9, R8, R10, R9 ;',
                       '/*01c0*/' + I + 'FADD R9, R8, R9 ;'),
 'w3_acc_init':       ('/*0090*/' + I + 'IMAD.MOV.U32 R9, RZ, RZ, RZ ;',
                       '/*0090*/' + I + 'IMAD.MOV.U32 R9, RZ, RZ, 0x1 ;'),
 'w7_acc_init_k0':    ('/*0090*/' + I + 'IMAD.MOV.U32 R9, RZ, RZ, RZ ;',
                       '/*0090*/' + I + 'IMAD.MOV.U32 R9, RZ, RZ, 0x3f800000 ;\n'
                       '        /*0098*/' + I + 'FSEL R9, RZ, R9, P0 ;'),
 'w4_top_guard':      ('/*0010*/' + I + 'ISETP.LT.U32.AND P0, PT, RZ, c[0x0][0x178], PT ;',
                       '/*0010*/' + I + 'ISETP.LT.U32.AND P0, PT, RZ, c[0x0][0x17c], PT ;'),
 'w5_inner_guard':    ('/*0080*/' + I + 'ISETP.LT.U32.AND P0, PT, RZ, c[0x0][0x180], PT ;',
                       '/*0080*/' + I + 'ISETP.LT.U32.AND P0, PT, RZ, c[0x0][0x17c], PT ;'),
 'w6_inner_backedge': ('/*01a0*/' + I + 'ISETP.GE.U32.AND P0, PT, R6, c[0x0][0x180], PT ;',
                       '/*01a0*/' + I + 'ISETP.GE.U32.AND P0, PT, R6, c[0x0][0x17c], PT ;'),
}
for name, (a, b) in twins.items():
    if src.count(a) != 1:
        raise SystemExit(f'y_cpu_matmul -O1 output moved: anchor for {name} found {src.count(a)} times')
    open(f'{d}/y_cpu_matmul_{name}.sass', 'w').write(src.replace(a, b))
    open(f'{d}/y_cpu_matmul_{name}.ptx', 'w').write(ptx)
print(f'  o1/y_cpu_matmul: {len(twins)} wrong twins derived')
PY
fi

# naive_gemm_f32 is the other -O1 subject, and it is a TRIPLE rather than a
# kernel.  The shipped PTX now says `fma.rn.f32` -- one rounding, which is what
# the hardware performs -- and it VALIDATES.  Both other readings of the same
# arithmetic are derived from it, because the difference between the three is
# the result:
#   _muladd  the form Y shipped until the emitter learned to say `fma`:
#            `mul.f32` then `add.f32`, two roundings, which ptxas contracts
#            into one FFMA.  It is REFUTED, and it is kept for that reason --
#            a validator whose corpus contains nothing it refutes reports
#            VALIDATED for a living and cannot be distinguished from one that
#            always does.
#   _rn      FORBID the contraction (mul.rn + add.rn).  Validates.  It is the
#            arm that needs FADD commutativity, because ptxas SORTS the addends
#            -- so it is also what keeps that identification exercised.
if [ -f "$REPO/tests/naive_gemm_f32.ptx" ]; then
  arch=$(grep -m1 -oE '^\.target[[:space:]]+sm_[0-9]+[a-z]*' "$REPO/tests/naive_gemm_f32.ptx" | awk '{print $2}')
  cp "$REPO/tests/naive_gemm_f32.ptx" o1/naive_gemm_f32.ptx
  # The rewrite FAILS LOUDLY if the emitter stops producing this shape, rather
  # than silently copying an unchanged file and reporting three identical arms.
  python3 - o1/naive_gemm_f32.ptx o1/naive_gemm_f32_muladd.ptx o1/naive_gemm_f32_rn.ptx <<'PY' || exit 1
import sys, re
src = open(sys.argv[1]).read()
m = re.search(r'^(\s*)fma\.rn\.f32 (%f\d+), (%f\d+), (%f\d+), (%f\d+);\n', src, re.M)
if not m:
    sys.stderr.write('  FAIL: naive_gemm_f32.ptx no longer has the fma.rn.f32 '
                     'shape this triple is about\n'); sys.exit(1)
ind, dst, a, b, acc = m.groups()
# Splitting one instruction into two needs one more register than the shipped
# kernel declares, and a body that names a register outside the declared pool
# is exactly the bug this directory's own history records.
rm = re.search(r'^(\s*)\.reg \.f32 %f<(\d+)>;', src, re.M)
if not rm:
    sys.stderr.write('  FAIL: naive_gemm_f32.ptx declares no .f32 register pool\n'); sys.exit(1)
n = int(rm.group(2)); prod = f'%f{n}'
grown = src[:rm.start()] + f'{rm.group(1)}.reg .f32 %f<{n+1}>;' + src[rm.end():]
d = len(grown) - len(src)                      # the splice shifted everything after it
def sub(rep): return grown[:m.start()+d] + rep + grown[m.end()+d:]
# (1) The form Y used to ship: two roundings, which ptxas fuses into one.
open(sys.argv[2], 'w').write(sub(
    f'{ind}mul.f32 {prod}, {a}, {b};\n{ind}add.f32 {dst}, {acc}, {prod};\n'))
# (2) FORBID the fusion: two roundings, and ptxas may not remove either.
open(sys.argv[3], 'w').write(sub(
    f'{ind}mul.rn.f32 {prod}, {a}, {b};\n{ind}add.rn.f32 {dst}, {acc}, {prod};\n'))
PY
  for v in naive_gemm_f32 naive_gemm_f32_muladd naive_gemm_f32_rn; do
    ptxas -O1 -arch="$arch" -o o1/$v.cubin o1/$v.ptx 2>/dev/null \
      && nvdisasm -c o1/$v.cubin > o1/$v.sass \
      && echo "  o1/$v rebuilt at -O1"
  done
fi

# ptx_integer_ops is the u32 DIVISION subject (divest.py, intenc.py): the only
# corpus kernel whose SASS carries ptxas's div/rem lowering and nothing past it.
# Its twins are derived from the genuine output, each by ONE asserted
# substitution, into idiv/ (not corpus/, which the census counts):
#   no_second_corr   the quotient's second correction is gone      -> refuted: e2 = I-1 needs it
#   rem_wrong        the remainder's second correction subtracts 0  -> refuted
#   rem_any_at_d0    the d == 0 select is unpredicated, so the rem is a
#                    different value ONLY at d == 0                -> VALIDATED: the PTX
#                    ISA leaves x % 0 unspecified (the positive control for that)
#   bias_moved       the estimate's bit-pattern bias is one larger  -> REFUSED by name:
#                    the device facts were measured for 0x0ffffffe
if [ -f "$OUT/ptx_integer_ops.sass" ]; then
  mkdir -p idiv
  python3 - "$OUT" <<'PY' || exit 1
import sys
d = sys.argv[1]
src = open(f'{d}/ptx_integer_ops.sass').read()
ptx = open(f'{d}/ptx_integer_ops.ptx').read()
I = '               '
twins = {
 'no_second_corr': (I + '@P2 IADD3 R4, R4, 0x1, RZ ;', I + '    NOP ;'),
 'rem_wrong':      ('IMAD.IADD R13, R14.reuse, 0x1, -R5 ;', 'IMAD.IADD R13, R14.reuse, 0x1, RZ ;'),
 'rem_any_at_d0':  (I + '@P3 SEL R3, R13, R14, P2 ;', I + '    SEL R3, R13, R14, P2 ;'),
 'bias_moved':     ('IADD3 R8, R10, 0xffffffe, RZ ;', 'IADD3 R8, R10, 0xfffffff, RZ ;'),
}
for name, (a, b) in twins.items():
    if src.count(a) != 1:
        raise SystemExit(f'ptx_integer_ops output moved: anchor for {name} found {src.count(a)} times')
    open(f'idiv/{name}.sass', 'w').write(src.replace(a, b))
    open(f'idiv/{name}.ptx', 'w').write(ptx)
print(f'  idiv/: {len(twins)} twins of ptx_integer_ops derived')
PY
fi

echo "corpus: $ok kernels, $skip skipped"
