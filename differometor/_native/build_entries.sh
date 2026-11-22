#!/bin/sh
# Builds libentries.so next to this script (the component formulas with derivatives, entries.cpp).
# The binary is committed; this script is only needed after a change of entries.cpp.
# Baseline x86-64 code and no fused multiply-add, as for libpattern_lu.so: the same results on
# every x86-64 CPU. No -ffast-math: nan and inf behave as in NumPy.
set -e
cd "$(dirname "$0")"
${CXX:-g++} -O2 -fPIC -shared -std=c++17 -ffp-contract=off -fno-math-errno -fno-exceptions -fno-rtti -Wall -Wextra \
    -o libentries.so entries.cpp
echo "built $(pwd)/libentries.so"
