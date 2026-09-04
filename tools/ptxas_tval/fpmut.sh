#!/bin/bash
cd "$(dirname "$0")"
./mkbase.sh fp_base.tgz || exit 1
run(){ timeout 60 python3 -u tval.py fma/rn.ptx fma/rn.sass 12 3 15 2>&1|tail -1; }
restore(){ rm -rf __pycache__; tar xzf fp_base.tgz; touch *.py fma/*; }
probe(){ restore; python3 -c "$2" || { echo "$1: APPLY FAILED"; return; }
         rm -rf __pycache__
         printf '%-46s ' "$1"; run; }

rm -f fp_base.tgz     # a surviving archive means this run did not finish
