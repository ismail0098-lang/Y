#!/bin/bash
# Restore + KILL THE BYTECODE CACHE.  `touch` is not enough: mtime granularity
# is one second, and a .pyc from a mutated import has already been served across
# a restore in this project and made a whole mutation table wrong.
cd "$(dirname "$0")"
./mkbase.sh guard_base.tgz || exit 1
n=$(tar tzf guard_base.tgz 2>/dev/null | wc -l)
[ "$n" -ge 3 ] || { echo "ARCHIVE HAS $n ENTRIES -- REFUSING TO RESTORE"; exit 1; }
tar xzf guard_base.tgz && rm -rf __pycache__ && echo "restored $n files, __pycache__ removed"
