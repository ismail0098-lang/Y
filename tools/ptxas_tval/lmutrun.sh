#!/bin/bash
cd "$(dirname "$0")"
./lmut.sh > lmut.log 2>&1
./regress.sh > regress.log 2>&1
echo done > lmut.done
