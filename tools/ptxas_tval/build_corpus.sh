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
rm -rf "$OUT" o1
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

# naive_gemm_f32 is the other -O1 subject, and it is a TRIPLE rather than a
# kernel: the shipped PTX says `mul.f32` then `add.f32` -- two roundings -- and
# ptxas contracts them into one FFMA, so the artifact does not mean what the
# machine does and the validator refutes it.  There are two repairs and the
# difference between them is the result, so both are built:
#   _rn   FORBID the contraction (mul.rn + add.rn).  Validates.  It is the arm
#         that needs FADD commutativity, because ptxas SORTS the addends -- so
#         it is also what keeps that identification exercised rather than dead.
#   _fma  SAY the contraction (fma.rn.f32).  Validates, and emits a
#         BYTE-IDENTICAL instruction stream to the shipped kernel.
if [ -f "$REPO/tests/naive_gemm_f32.ptx" ]; then
  arch=$(grep -m1 -oE '^\.target[[:space:]]+sm_[0-9]+[a-z]*' "$REPO/tests/naive_gemm_f32.ptx" | awk '{print $2}')
  cp "$REPO/tests/naive_gemm_f32.ptx" o1/naive_gemm_f32.ptx
  # The rewrite FAILS LOUDLY if the emitter stops producing this shape, rather
  # than silently copying an unchanged file and reporting two identical halves.
  python3 - o1/naive_gemm_f32.ptx o1/naive_gemm_f32_rn.ptx o1/naive_gemm_f32_fma.ptx <<'PY' || exit 1
import sys, re
src = open(sys.argv[1]).read()
m = re.search(r'^(\s*)mul\.f32 (%f\d+), (%f\d+), (%f\d+);\n\s*add\.f32 (%f\d+), (%f\d+), \2;\n',
              src, re.M)
if not m:
    sys.stderr.write('  FAIL: naive_gemm_f32.ptx no longer has the mul.f32+add.f32 '
                     'shape this triple is about\n'); sys.exit(1)
ind, prod, a, b, dst, acc = m.groups()
# (1) FORBID the contraction: two roundings, as the shipped PTX literally says.
open(sys.argv[2], 'w').write(src[:m.start()] +
    f'{ind}mul.rn.f32 {prod}, {a}, {b};\n{ind}add.rn.f32 {dst}, {acc}, {prod};\n' +
    src[m.end():])
# (2) SAY the contraction: one rounding, which is what the machine does anyway.
open(sys.argv[3], 'w').write(src[:m.start()] +
    f'{ind}fma.rn.f32 {dst}, {a}, {b}, {acc};\n' + src[m.end():])
PY
  for v in naive_gemm_f32 naive_gemm_f32_rn naive_gemm_f32_fma; do
    ptxas -O1 -arch="$arch" -o o1/$v.cubin o1/$v.ptx 2>/dev/null \
      && nvdisasm -c o1/$v.cubin > o1/$v.sass \
      && echo "  o1/$v rebuilt at -O1"
  done
fi

echo "corpus: $ok kernels, $skip skipped"
