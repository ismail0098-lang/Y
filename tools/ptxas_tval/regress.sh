#!/bin/bash
# Standing results.  Run from a FILE, one target at a time, never concurrently.
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
