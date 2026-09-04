#!/bin/bash
# one probe: restore, apply, VERIFY IT LANDED, run
cd "$(dirname "$0")"
./mkbase.sh smem_base.tgz || exit 1
tar xzf smem_base.tgz && rm -rf __pycache__
[ -n "$1" ] && python3 -c "$1"
if [ -n "$2" ]; then grep -q "$2" $3 || { echo "MUTATION DID NOT LAND in $3"; exit 9; }; fi
timeout 600 python3 smemval.py smut/smem_roundtrip.ptx smut/smem_roundtrip.sass 60 wide 2>&1 | tail -1
