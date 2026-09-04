#!/bin/bash
cd "$(dirname "$0")"
./mkbase.sh smem_base.tgz || exit 1
declare -A D=(
 [S0]="CONTROL: reorder the two alignment lists (no-op)"
 [S1]="barrier modelled as a NO-OP, kernel unchanged"
 [S1b]="no-op barrier AND the store moved across it"
 [S1c]="store moved across the barrier, REAL barrier model"
 [S2]=".X16 address scaling dropped"
 [S3]="shared window: shift-then-truncate (today's bug)"
 [S4]="SASS shared store deleted"
 [S5]="address match back to unconditional"
 [S6]="a barrier deleted from the SASS"
 [S7]="shared symbol base resolved to 64, not 0"
 [S8]="a LOAD moved across the barrier, REAL model"
 [S8b]="a LOAD moved across the barrier, NO-OP barrier"
)
printf '%-6s %-52s %s\n' MUT DESCRIPTION 'smem_roundtrip'
for m in S0 S1 S1b S1c S2 S3 S4 S5 S6 S7 S8 S8b; do
  tar xzf smem_base.tgz && rm -rf __pycache__
  python3 "muts/$m.py" 2>/dev/null || { printf '%-6s %-52s %s\n' "$m" "${D[$m]}" "MUTATION FAILED TO APPLY"; continue; }
  R=$(timeout 600 python3 smemval.py smut/smem_roundtrip.ptx smut/smem_roundtrip.sass 60 wide 2>&1 | tail -1)
  [ -z "$R" ] && R="NO RESULT LINE = FAILURE"
  printf '%-6s %-52s %s\n' "$m" "${D[$m]}" "$R"
done
tar xzf smem_base.tgz && rm -rf __pycache__
printf '\n%-6s %-52s %s\n' RESTORED "BASELINE" "$(timeout 600 python3 smemval.py smut/smem_roundtrip.ptx smut/smem_roundtrip.sass 60 wide 2>&1 | tail -1)"

rm -f smem_base.tgz   # a surviving archive means this run did not finish
