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
            "neg/folded.ptx neg/folded.sass" "neg/sub.ptx neg/sub.sass" \
            "neg/unfoldable.ptx neg/unfoldable.sass" \
            "max/relu.ptx max/relu.sass" "max/general.ptx max/general.sass" \
            "max/min.ptx max/min.sass" \
            "corpus/bn254_permute.ptx corpus/bn254_permute.sass" \
            "corpus/bn254_sub_vec.ptx corpus/bn254_sub_vec.sass" \
            "corpus/ptx_carry_chain.ptx corpus/ptx_carry_chain.sass"; do
  set -- $pair
  name=$(basename "$1" .ptx)
  out=$(timeout 600 python3 tval.py "$1" "$2" 12 3 15 2>&1 | tail -1)
  printf '%-22s %s\n' "$name" "$out"
  # `fma/plain` is the NEGATIVE CONTROL and is asserted in its own direction:
  # every result above is worth exactly what that row is worth.
  #
  # `max/relu` is the SHIPPED ReLU epilogue's shape and `max/general` is the
  # same opcode on two runtime values.  They are NOT a fixture and a spare: the
  # ReLU form carries a literal, which ptxas folds into RZ and puts in the FIRST
  # operand slot, so `relu` needs FMAX commutativity and `general` -- whose
  # operand order ptxas preserves -- validates without it.  Measured: with the
  # canonicalisation removed, relu goes UNPROVED and general stays VALIDATED.
  # A general-max fixture alone would have hidden that the fact is needed.
  #
  # `max/min` is the OTHER polarity of the same SASS instruction, so the two of
  # them pin that the executor reads FMNMX's predicate operand rather than
  # assuming a polarity.
  #
  # `neg/unfoldable` is the SECOND negative control and it is a different
  # refutation from `plain`: not a contraction ptxas is free to make, but a
  # `neg.f32` whose un-foldable lowering (`FADD Rd, -Rx, -RZ`) is arithmetic and
  # canonicalises every NaN.  Its sibling `neg/folded` is the SAME PTX opcode in
  # the shape the whole corpus uses, and it VALIDATES -- so the pair pins that
  # the refusal is about the lowering rather than about the opcode.
  case "$name" in
    plain|unfoldable) echo "$out" | grep -q '^UNPROVED'  || bad=$((bad+1)) ;;
    *)                echo "$out" | grep -q '^VALIDATED' || bad=$((bad+1)) ;;
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
# The GLOBAL MEMORY MODEL's preconditions (memorder.py).  Every row below except
# the two controls is a program the validator VALIDATED before memorder.py
# existed, and six of the nine are WRONG translations built by hand from
# ptxas's own output -- see each .ptx header.  The directions are the result:
#
#   REFUSED   las_sass las_ptx   a load that could read back a store, one row per
#                                side, so neither half of the check can go alone
#   REFUSED   lsls               the CORRECT ptxas output for las_ptx, and the
#                                price of refusing: the model cannot tell it from
#                                the wrong one.  A store-ordered memory model is
#                                what turns this row green while las_ptx stays red
#   UNPROVED  swap_alias swap_off3   stores reordered across a possible overlap
#   VALIDATED swap_off4 off3     the controls: a reorder that cannot overlap, and
#                                overlapping stores in the SAME order
#   REFUSED   pstore pstore_wrong    a store BEFORE a loop, which loopval never
#                                counted -- the wrong one wrote a different value
#   VALIDATED pret               a predicated early return, CORRECT -- UNPROVED
#                                before, because ptxexec read `ret` as `pass`
#   UNPROVED  pret_wrong         the same with the SASS EXIT deleted -- VALIDATED
#                                before: the pair's verdicts were inverted
for n in las_sass las_ptx lsls swap_alias swap_off3 swap_off4 off3 pret pret_wrong; do
  out=$(timeout 300 python3 tval.py "mem/$n.ptx" "mem/$n.sass" 12 3 15 2>&1 | tail -1)
  printf '%-22s %s\n' "mem/$n" "$out"
  case "$n" in
    swap_off4|off3|pret)  echo "$out" | grep -q '^VALIDATED' || bad=$((bad+1)) ;;
    pret_wrong)           echo "$out" | grep -q '^UNPROVED.*guard' || bad=$((bad+1)) ;;
    swap_alias|swap_off3) echo "$out" | grep -q '^UNPROVED.*REORDERED' || bad=$((bad+1)) ;;
    *)                    echo "$out" | grep -q '^REFUSED.*read back' || bad=$((bad+1)) ;;
  esac
done
for n in pstore pstore_wrong; do
  out=$(timeout 300 python3 loopval.py "mem/$n.ptx" "mem/$n.sass" 60 wide 2>&1 | tail -1)
  printf '%-22s %s\n' "mem/$n" "$out"
  echo "$out" | grep -q '^REFUSED.*store in the SASS prologue\|^REFUSED.*store in the PTX prologue' || bad=$((bad+1))
done
# The LOOP validator's other sites.  Each `_wrong`/`_exit`/`_nostore` row was
# VALIDATED by loopval before this change; `loop_swap` is their control and
# `loop_ls` is a correct translation refused only because the body is run twice.
for n in loop_ls loop_swap loop_swap_wrong loop_swap_exit loop_body_exit loop_nostore loop_ret_wrong; do
  out=$(timeout 300 python3 loopval.py "mem/$n.ptx" "mem/$n.sass" 60 wide 2>&1 | tail -1)
  printf '%-22s %s\n' "mem/$n" "$out"
  case "$n" in
    loop_ls)         echo "$out" | grep -q '^REFUSED.*read back'        || bad=$((bad+1)) ;;
    loop_swap)       echo "$out" | grep -q '^VALIDATED'                 || bad=$((bad+1)) ;;
    loop_swap_wrong) echo "$out" | grep -q '^UNPROVED.*REORDERED'       || bad=$((bad+1)) ;;
    loop_nostore)    echo "$out" | grep -q '^REFUSED.*stores nothing'   || bad=$((bad+1)) ;;
    *)               echo "$out" | grep -q '^REFUSED.*can end the program' || bad=$((bad+1)) ;;
  esac
done
# THE THIRD VALIDATOR.  `batch.validate` pairs global stores too, and its only
# standing caller is `smemval` on a kernel with no loads -- so without these
# rows its two new checks would be reached by nothing.  A guard consulted at
# two of three sites is the bug this directory keeps finding.
for n in las_sass las_ptx swap_alias; do
  out=$(timeout 300 python3 smemval.py "mem/$n.ptx" "mem/$n.sass" 60 wide 2>&1 | tail -1)
  printf '%-22s %s\n' "smemval mem/$n" "$out"
  case "$n" in
    swap_alias) echo "$out" | grep -q '^UNPROVED.*REORDERED' || bad=$((bad+1)) ;;
    *)          echo "$out" | grep -q '^REFUSED.*read back'  || bad=$((bad+1)) ;;
  esac
done
if [ "$bad" -ne 0 ]; then echo; echo "FAIL: $bad standing result(s) moved."; exit 1; fi
