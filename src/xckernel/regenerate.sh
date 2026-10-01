#!/bin/sh
# Regenerate the vendored libxckernel tree in this directory.
#
# libxckernel (https://github.com/susilehtola/libxckernel) generates the
# exchange-correlation kernel contractions that ERKALE uses: the XC
# energy (order 0), Fock matrix (order 1) and response (order 2) of
# every functional family, including the current-corrected meta-GGAs.
# The generated C/C++ sources are committed here, so that building
# ERKALE does not need Python. This script reproduces them exactly from
# the pinned libxckernel commit; change the commit or the selection
# below and rerun it to update them.
#
# Needs git and Python >= 3.10 with sympy and numpy.
set -e

XCKERNEL_REPO=${XCKERNEL_REPO:-https://github.com/susilehtola/libxckernel.git}
XCKERNEL_COMMIT=${XCKERNEL_COMMIT:-59f3353623a92f3a93fcbce2facfeeaa3e537002}
FAMILIES=lda,gga,mgga_tau,mgga_lapl,mgga,cmgga_tau
MAX_ORDER=2

here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

git clone --quiet "$XCKERNEL_REPO" "$work/libxckernel"
git -C "$work/libxckernel" checkout --quiet "$XCKERNEL_COMMIT"
(cd "$work/libxckernel" && python3 -m xckernel.catalog "$work/out" "$FAMILIES" "$MAX_ORDER" c)
cp "$work/libxckernel/LICENSE" "$work/out/LICENSE"

# Replace everything generated, keeping this script and the README
find "$here" -mindepth 1 -maxdepth 1 ! -name regenerate.sh ! -name README.md -exec rm -rf {} +
cp -R "$work/out/." "$here/"
printf '%s\n' "$XCKERNEL_COMMIT" > "$here/XCKERNEL_COMMIT"
