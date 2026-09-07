#!/bin/bash
# Standing results.  Run from a FILE, one target at a time, never concurrently.
#
# ALL of them: the straight-line cases through tval.py, and the loop and
# shared-memory cases that the README used to list as three commands a reader
# had to type.  A documented command nothing runs is how a result goes stale.
cd "$(dirname "$0")"
rm -rf __pycache__

bad=0
for pair in "fma/rn.ptx fma/rn.sass" "fma/plain.ptx fma/plain.sass" \
            "corpus/bn254_permute.ptx corpus/bn254_permute.sass" \
            "corpus/bn254_sub_vec.ptx corpus/bn254_sub_vec.sass" \
            "corpus/ptx_carry_chain.ptx corpus/ptx_carry_chain.sass"; do
  set -- $pair
  name=$(basename "$1" .ptx)
  out=$(timeout 600 python3 tval.py "$1" "$2" 12 3 15 2>&1 | tail -1)
  printf '%-22s %s\n' "$name" "$out"
  # `fma/plain` is the NEGATIVE CONTROL and is asserted in its own direction:
  # every result above is worth exactly what that row is worth.
  case "$name" in
    plain) echo "$out" | grep -q '^UNPROVED'  || bad=$((bad+1)) ;;
    *)     echo "$out" | grep -q '^VALIDATED' || bad=$((bad+1)) ;;
  esac
done

# Loop and shared memory.  naive_gemm_f32 is a TRIPLE and the difference
# between its arms is the result.  The SHIPPED kernel says `fma.rn.f32` and
# validates.  `_muladd` is the form Y shipped before the emitter learned to say
# that -- `mul.f32` then `add.f32`, which ptxas contracts into one FFMA, so the
# artifact does not mean what the machine does -- and it is REFUTED on a
# BYTE-IDENTICAL instruction stream.  `_rn` forbids the fusion instead and
# validates, at a different instruction stream.
#
# The UNPROVED row is a standing result too, and a run in which it turns green
# is a regression: a corpus containing nothing the validator refutes cannot be
# told apart from a validator that always says VALIDATED.
for t in "loopval o1/exact_pv" "loopval o1/naive_gemm_f32" \
         "loopval o1/naive_gemm_f32_muladd" "loopval o1/naive_gemm_f32_rn" \
         "smemval smut/smem_roundtrip"; do
  set -- $t
  name=$(basename "$2")
  out=$(timeout 900 python3 "$1.py" "$2.ptx" "$2.sass" 60 wide 2>&1 | tail -1)
  printf '%-22s %s\n' "$name" "$out"
  # ASSERT, do not merely print.  Every row here is a standing result in the
  # direction it currently reads -- `_muladd` turning green is a regression
  # exactly as much as a VALIDATED row turning red, because a corpus that
  # refutes nothing cannot be told apart from a validator that never refutes.
  case "$name" in
    naive_gemm_f32_muladd) echo "$out" | grep -q '^UNPROVED'  || bad=$((bad+1)) ;;
    *)                     echo "$out" | grep -q '^VALIDATED' || bad=$((bad+1)) ;;
  esac
done
if [ "$bad" -ne 0 ]; then echo; echo "FAIL: $bad standing result(s) moved."; exit 1; fi
