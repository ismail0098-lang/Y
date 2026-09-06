#!/bin/bash
# Standing results.  Run from a FILE, one target at a time, never concurrently.
#
# ALL of them: the straight-line cases through tval.py, and the loop and
# shared-memory cases that the README used to list as three commands a reader
# had to type.  A documented command nothing runs is how a result goes stale.
cd "$(dirname "$0")"
rm -rf __pycache__

for pair in "fma/rn.ptx fma/rn.sass" "fma/plain.ptx fma/plain.sass" \
            "corpus/bn254_permute.ptx corpus/bn254_permute.sass" \
            "corpus/bn254_sub_vec.ptx corpus/bn254_sub_vec.sass" \
            "corpus/ptx_carry_chain.ptx corpus/ptx_carry_chain.sass"; do
  set -- $pair
  name=$(basename "$1" .ptx)
  out=$(timeout 600 python3 tval.py "$1" "$2" 12 3 15 2>&1 | tail -1)
  printf '%-22s %s\n' "$name" "$out"
done

# Loop and shared memory.  naive_gemm_f32 is a PAIR and the difference between
# its halves is the result: the shipped PTX asks for two roundings, ptxas
# contracts them into one FFMA, and the same kernel saying `fma.rn.f32` emits a
# BYTE-IDENTICAL instruction stream and validates.  So the UNPROVED row is a
# standing result too, and a run in which it turns green is a regression.
for t in "loopval o1/exact_pv" "loopval o1/naive_gemm_f32" \
         "loopval o1/naive_gemm_f32_rn" "loopval o1/naive_gemm_f32_fma" \
         "smemval smut/smem_roundtrip"; do
  set -- $t
  name=$(basename "$2")
  out=$(timeout 900 python3 "$1.py" "$2.ptx" "$2.sass" 60 wide 2>&1 | tail -1)
  printf '%-22s %s\n' "$name" "$out"
done
