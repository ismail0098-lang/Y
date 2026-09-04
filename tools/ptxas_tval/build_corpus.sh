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

echo "corpus: $ok kernels, $skip skipped"
