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
# The GLOBAL MEMORY MODEL (memorder.py).  Every row below except the controls is
# a program the validator VALIDATED before memorder.py existed, and six of the
# nine are WRONG translations built by hand from ptxas's own output -- see each
# .ptx header.  The directions are the result:
#
#   UNPROVED  las_sass las_ptx   a load that could read back a store, one row per
#                                side: REFUTED with a counterexample (`store 1:
#                                sat`) now that a load reads through the stores
#                                before it.  They were REFUSED while the model
#                                read every load from the initial array; asserting
#                                the `sat` is what separates a refutation from a
#                                solver timeout
#   VALIDATED lsls               the CORRECT ptxas output for las_ptx.  It was the
#                                refusal's price -- the old model could not tell
#                                it from the wrong one -- and turning it green
#                                while las_ptx stays refuted is what the
#                                store-ordered model is for
#   UNPROVED  swap_alias swap_off3   stores reordered across a possible overlap
#   VALIDATED swap_off4 off3     the controls: a reorder that cannot overlap, and
#                                overlapping stores in the SAME order
#   REFUSED   pstore pstore_wrong    a store BEFORE a loop, which loopval never
#                                counted -- the wrong one wrote a different value
#   VALIDATED sls_alias          two stores to ONE address with a possible read-back
#                                between them, ptxas's genuine output.  It pins the
#                                ORDER memorder.pair_by_address pairs a same-address
#                                group in: reversed, this correct row goes UNPROVED,
#                                and no other row has two stores at one address
#   VALIDATED pret               a predicated early return, CORRECT -- UNPROVED
#                                before, because ptxexec read `ret` as `pass`
#   UNPROVED  pret_wrong         the same with the SASS EXIT deleted -- VALIDATED
#                                before: the pair's verdicts were inverted
for n in las_sass las_ptx lsls swap_alias swap_off3 swap_off4 off3 sls_alias pret pret_wrong; do
  full=$(timeout 300 python3 tval.py "mem/$n.ptx" "mem/$n.sass" 12 3 15 2>&1)
  out=$(echo "$full" | tail -1)
  printf '%-22s %s\n' "mem/$n" "$out"
  case "$n" in
    # lsls at EXACTLY 10: the eleventh obligation is the direct-multiply refinement,
    # which runs only when the abstraction cannot discharge the read-through --
    # measured 8-11 s without the address hook, 0.03 s with it.  The count is the
    # structural form of that performance property; a timing would be flaky.
    lsls)                 echo "$out" | grep -q '^VALIDATED  10 obligations' || bad=$((bad+1)) ;;
    swap_off4|off3|sls_alias|pret) echo "$out" | grep -q '^VALIDATED' || bad=$((bad+1)) ;;
    pret_wrong)           echo "$out" | grep -q '^UNPROVED.*guard' || bad=$((bad+1)) ;;
    swap_alias|swap_off3) echo "$out" | grep -q '^UNPROVED.*REORDERED' || bad=$((bad+1)) ;;
    las_sass|las_ptx)     { echo "$out" | grep -q '^UNPROVED' && echo "$full" | grep -q '^  store 1: sat$'; } || bad=$((bad+1)) ;;
    *)                    bad=$((bad+1)) ;;
  esac
done
for n in pstore pstore_wrong; do
  out=$(timeout 300 python3 loopval.py "mem/$n.ptx" "mem/$n.sass" 60 wide 2>&1 | tail -1)
  printf '%-22s %s\n' "mem/$n" "$out"
  echo "$out" | grep -q '^REFUSED.*store in the SASS prologue\|^REFUSED.*store in the PTX prologue' || bad=$((bad+1))
done
# SUB-WORD STORES AND A REAL READ-BACK.  `ptx_subword_ops` is a committed corpus
# kernel: its PTX loads A8 again AFTER a byte store to OBack8, and it stores
# sub-word values.  It was blocked on three opcodes the old memory model could
# not make sound (it had no store width) and on a read-back it refused.  It
# VALIDATES now, and its two wrong twins are each asserted in their own direction
# -- a corpus row that turns green is worth what its refutations are worth:
#   subword_hoist  the SASS read-back hoisted above the byte store: REFUTED
#   subword_widen  the byte store widened to a word: UNPROVED on the store width
# The device facts under it (a sub-word store writes exactly its bytes; u8
# conversion truncates; wide accesses fault misaligned) are `subword_abi.py`.
for n in corpus/ptx_subword_ops mem/subword_hoist mem/subword_widen; do
  full=$(timeout 900 python3 tval.py "$n.ptx" "$n.sass" 12 3 15 2>&1)
  out=$(echo "$full" | tail -1)
  printf '%-22s %s\n' "$n" "$out"
  case "$n" in
    corpus/ptx_subword_ops) echo "$out" | grep -q '^VALIDATED' || bad=$((bad+1)) ;;
    mem/subword_hoist)      { echo "$out" | grep -q '^UNPROVED' && echo "$full" | grep -q '^  store 6: sat$'; } || bad=$((bad+1)) ;;
    mem/subword_widen)      echo "$out" | grep -q '^UNPROVED.*store 5 width: ptx 8 bits, sass 32 bits' || bad=$((bad+1)) ;;
  esac
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
#
# `lsls` is here too: the address hook that makes a read-through cheap is wired
# into this validator separately (`batch.validate` builds its own abstract
# posing), so its correct read-back needs a row of its own or that wiring is
# reached by nothing.
for n in las_sass las_ptx swap_alias lsls; do
  out=$(timeout 300 python3 smemval.py "mem/$n.ptx" "mem/$n.sass" 60 wide 2>&1 | tail -1)
  printf '%-22s %s\n' "smemval mem/$n" "$out"
  case "$n" in
    swap_alias) echo "$out" | grep -q '^UNPROVED.*REORDERED'        || bad=$((bad+1)) ;;
    lsls)       echo "$out" | grep -q '^VALIDATED'                  || bad=$((bad+1)) ;;
    *)          echo "$out" | grep -q '^UNPROVED.*store 1 value: sat' || bad=$((bad+1)) ;;
  esac
done
# THE NEST VALIDATOR (nestval.py).  `y_cpu_matmul` at -O1 is three nested loops
# with a store in the middle one -- the kernel the multi-back-edge lift was
# priced on -- and it VALIDATES at exactly 17 obligations.  Every row after it
# is asserted in its OWN direction, because a nest validator that always said
# VALIDATED would report that row identically:
#   w1..w6          one wrong instruction each, refuted at the obligation that
#                   mutation breaks (store address, store value, BASE, two
#                   ENTRY guards, LOOPCOND)
#   nest_accum      b[0] += a[i]: a read-back ACROSS iterations, VALIDATED
#   nest_accum_stale  the same with b[0] loaded once before the loop: refuted,
#                   and only a model carrying memory between iterations can see it
#   _rn, _muladd, exact_pv   agreement with loopval on single loops
#   loop_swap_exit  an EXIT zero-trip guard with stores after the nest: refused
for t in o1/y_cpu_matmul o1/y_cpu_matmul_w1_store_stride o1/y_cpu_matmul_w2_acc_add \
         o1/y_cpu_matmul_w3_acc_init o1/y_cpu_matmul_w4_top_guard o1/y_cpu_matmul_w5_inner_guard \
         o1/y_cpu_matmul_w6_inner_backedge o1/y_cpu_matmul_w7_acc_init_k0 mem/nest_accum mem/nest_accum_stale \
         o1/naive_gemm_f32_rn o1/naive_gemm_f32_muladd o1/exact_pv mem/loop_ls mem/loop_swap_exit; do
  out=$(timeout 600 python3 nestval.py "$t.ptx" "$t.sass" 20 2>&1 | tail -1)
  printf '%-22s %s\n' "nest ${t#*/}" "$out"
  case "$t" in
    o1/y_cpu_matmul)             echo "$out" | grep -q '^VALIDATED  17 obligations'                 || bad=$((bad+1)) ;;
    *w1_store_stride)            echo "$out" | grep -q '^UNPROVED.*iteration store 0 address'       || bad=$((bad+1)) ;;
    *w2_acc_add)                 echo "$out" | grep -q '^UNPROVED.*iteration store 0 value: sat'    || bad=$((bad+1)) ;;
    *w3_acc_init|*w7_acc_init_k0) echo "$out" | grep -q '^UNPROVED.*iteration store 0 value: sat'   || bad=$((bad+1)) ;;
    *w4_top_guard|*w5_inner_guard) echo "$out" | grep -q '^UNPROVED.*ENTRY: zero-trip guards disagree' || bad=$((bad+1)) ;;
    *w6_inner_backedge)          echo "$out" | grep -q '^UNPROVED.*LOOPCOND'                        || bad=$((bad+1)) ;;
    mem/nest_accum_stale)        echo "$out" | grep -q '^UNPROVED.*iteration store 0 value: sat'    || bad=$((bad+1)) ;;
    o1/naive_gemm_f32_muladd)    echo "$out" | grep -q '^UNPROVED.*epilogue store 0 value: sat'     || bad=$((bad+1)) ;;
    mem/loop_swap_exit)          echo "$out" | grep -q '^REFUSED.*EXITs when the nest runs zero times' || bad=$((bad+1)) ;;
    *)                           echo "$out" | grep -q '^VALIDATED'                                 || bad=$((bad+1)) ;;
  esac
done
# A REFUSAL RAISED INSIDE AN EXECUTOR IS A REFUSAL, NOT A CRASH.  `memorder`
# can refuse while `ptxexec`/`sassexec` build the state, and `tval.run` let that
# escape as a traceback.  No fixture reaches it (a real kernel that does would
# be a refusal worth its own row), so the executor is made to raise and the
# row asserts `tval.run` names it -- through `run` itself, not a copy of it.
out=$(timeout 60 python3 -c "
import tval, ptxexec, memorder
def boom(*a, **k): raise memorder.Refusal('probe refusal from inside an executor')
ptxexec.run_ptx = boom
v, msg, n = tval.run('mem/lsls.ptx', 'mem/lsls.sass', log=lambda *a: None)
print(v, n, msg)" 2>&1 | tail -1)
printf '%-22s %s\n' "tval executor refusal" "$out"
echo "$out" | grep -q '^REFUSED 0 probe refusal from inside an executor' || bad=$((bad+1))
# THE u32 DIVISION LOWERING.  ptx_integer_ops was REFUSED on I2F.U32.RP, and
# its tail was `unknown` over bitvectors on six posings; divest.py models the
# estimate as a fresh value carrying the two device-measured facts
# (divlow_abi.py), and intenc.py discharges the tail over Int.  The twins are
# derived by build_corpus.sh and are asserted by the STORE that fails, since
# the verdict alone cannot tell the quotient from the remainder:
#   no_second_corr   refuted at store 3 (the quotient): e2 = I-1 needs it
#   rem_wrong        refuted at store 4 (the remainder)
#   rem_any_at_d0    VALIDATED: differs only at d == 0, where the PTX ISA says
#                    the result is unspecified -- the positive control for that
#   bias_moved       REFUSED by name: the facts were measured for 0x0ffffffe
#   udiv/twice       one division stored twice: VALIDATED
#   udiv/twice_split the second store differs from the first ONLY at d == 0:
#                    unspecified is ONE value, so this is refuted -- by the
#                    consistency check and by nothing else (no corpus kernel
#                    stores a quotient twice, so without this pair that check
#                    would be a guard nothing reaches)
#   udiv/and1        only the quotient's LOW BIT is stored; at d == 0 the spec
#                    allows 0 or 1 and nothing else, which tval cannot express,
#                    so the store is refused by name (conservatively UNPROVED)
#   udiv/and1_two    ...and made to store 2 at d == 0: the same refusal.  With
#                    it removed this twin is STILL unproved, because `(n/d) & 1`
#                    does not prove over Int at d != 0 either -- so the refusal
#                    is reached but not isolated, and the rows assert it FIRES
# The genuine row asserts it went through the Int rung, or a change that made
# it pass for another reason would read as this result.
for p in corpus/ptx_integer_ops idiv/no_second_corr idiv/rem_wrong idiv/rem_any_at_d0 idiv/bias_moved udiv/twice udiv/twice_split udiv/and1 udiv/and1_two; do
  full=$(timeout 900 python3 tval.py "$p.ptx" "$p.sass" 12 3 15 2>&1)
  out=$(echo "$full" | tail -1)
  printf '%-22s %s\n' "$(basename $p)" "$out"
  case "$p" in
    corpus/ptx_integer_ops) echo "$out" | grep -q '^VALIDATED .*3 over Int'                         || bad=$((bad+1)) ;;
    idiv/no_second_corr)    { echo "$out" | grep -q '^UNPROVED' && echo "$full" | grep -q 'store 3: sat'; } || bad=$((bad+1)) ;;
    idiv/rem_wrong)         { echo "$out" | grep -q '^UNPROVED' && echo "$full" | grep -q 'store 4: sat'; } || bad=$((bad+1)) ;;
    idiv/rem_any_at_d0)     echo "$out" | grep -q '^VALIDATED'                                      || bad=$((bad+1)) ;;
    idiv/bias_moved)        echo "$out" | grep -q '^REFUSED.*not the measured bias'                 || bad=$((bad+1)) ;;
    udiv/twice)             echo "$out" | grep -q '^VALIDATED'                                      || bad=$((bad+1)) ;;
    udiv/twice_split)       { echo "$out" | grep -q '^UNPROVED' && echo "$full" | grep -q 'same unspecified division'; } || bad=$((bad+1)) ;;
    udiv/and1|udiv/and1_two) { echo "$out" | grep -q '^UNPROVED' && echo "$full" | grep -q 'computed FROM an unspecified division'; } || bad=$((bad+1)) ;;
  esac
done
if [ "$bad" -ne 0 ]; then echo; echo "FAIL: $bad standing result(s) moved."; exit 1; fi
