#!/bin/bash
#
# Analytic validation of the per-state protonic moments and the state averaging
# in erkale_hneoci (V1-V6).  See README_MOMENTS.md.
#
# The electron is decoupled by giving it a single, very diffuse s primitive: its
# density is then flat over the proton's extent, so the proton feels the trap
# plus a constant.  Because that electronic density is spherical, the protonic
# CI states are *exactly* the harmonic-oscillator states -- the s/p block of the
# CI Hamiltonian is parity-diagonal -- and every reference below is exact.
#
# The protonic basis is a single s and a single p primitive at zeta = m w / 2,
# which represents the n=0 and n=1 oscillator states exactly.
#
# Usage:  tests/moment_tests.sh [path-to-erkale_hneoci]
# Env:    ERKALE_LIBRARY (basis directory) -- defaults to <repo>/basis
#
set -u

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
export ERKALE_LIBRARY=${ERKALE_LIBRARY:-$root/basis}

# Locate the hneoci executable (arg overrides autodetection).
HN=${1:-}
if [ -z "$HN" ]; then
  for c in "$root"/openmp/src/contrib/erkale_hneoci_omp \
           "$root"/serial/src/contrib/erkale_hneoci \
           "$(command -v erkale_hneoci_omp 2>/dev/null)" \
           "$(command -v erkale_hneoci 2>/dev/null)"; do
    [ -n "$c" ] && [ -x "$c" ] && HN="$c" && break
  done
fi
if [ -z "$HN" ] || [ ! -x "$HN" ]; then
  echo "ERROR: erkale_hneoci not found. Build it, or pass its path as argument."
  exit 2
fi
echo "Using binary : $HN"
echo "ERKALE_LIBRARY: $ERKALE_LIBRARY"

mp=1836.15267389            # must match ProtonMass default in hneoci.cpp
w=0.01                      # isotropic trap frequency, Hartree
lam=-0.02                   # cross-term coefficient used to resolve sigma_z
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# A single, very diffuse electronic s primitive: decouples the electron.
python3 -c "print('H     0');print('S   1   1.00');print('   %.10e   1.00'%1e-6);print('****')" > "$work/e.gbs"
# Protonic s + p primitives matched to the trap: zeta = m w / 2.
python3 -c "
z=$mp*$w/2
print('H     0')
for am in ('S','P'):
    print('%s   1   1.00'%am); print('   %.12f   1.00'%z)
print('****')" > "$work/p.gbs"

# run <tag> <extra keywords...>  -- writes <tag>.run, executes, keeps <tag>.out
run() {
  local tag=$1; shift
  { printf '%s\n' "$@"; } > "$work/$tag.run"
  if ! "$HN" "$work/$tag.run" > "$work/$tag.out" 2>&1; then
    echo "ERROR: $tag failed:"; tail -3 "$work/$tag.out"; exit 3
  fi
}

trapkw="NEOTrap true
TrapOmegaUnit Eh
TrapOmegaPar $w
TrapOmegaPerp $w"

# V6 -- the defaults must not print a root table nor write a sidecar.
run reg_default "Basis cc-pVDZ"
if [ -e "$work/reg_default.props.json" ]; then
  echo "FAIL V6: a sidecar was written for a default run"; exit 1
fi
if grep -qE "^ root|state-avg" "$work/reg_default.out"; then
  echo "FAIL V6: the default run printed the multi-root block"; exit 1
fi
echo "PASS V6 defaults print no multi-root block and write no sidecar"

# ... and the multi-root machinery must not perturb the CI energy.
run reg_multi "Basis cc-pVDZ" "NRoots 4" "PrintMoments true" "StateAverage true"

# V1, V3, V4, V5b, V5c: isotropic trap, equal weights over the n=1 manifold.
run v1 "Basis $work/e.gbs" "ProtonBasis $work/p.gbs" "$trapkw" \
       "NRoots 4" "PrintMoments true" "MomentsFull true" \
       "StateAverage true" "SAWeights 0.0 1.0 1.0 1.0" "SADensityOut true"

# V2: lam_cross < 0 splits sigma_z below the pi doublet without mixing states.
run v2 "Basis $work/e.gbs" "ProtonBasis $work/p.gbs" "$trapkw" \
       "TrapLambdaCross $lam" "NRoots 4" "PrintMoments true"

# V5a: all the weight on the ground root.
run v5a "Basis $work/e.gbs" "ProtonBasis $work/p.gbs" "$trapkw" \
        "NRoots 4" "PrintMoments true" "StateAverage true" "SAWeights 1.0 0.0 0.0 0.0"

# V5d: equal weights (empty SAWeights) give the arithmetic mean.
run v5d "Basis $work/e.gbs" "ProtonBasis $work/p.gbs" "$trapkw" \
        "NRoots 4" "PrintMoments true" "StateAverage true"

# The state-averaged density must still carry exactly one proton.
python3 - "$work" <<'PY'
import sys, json
import itertools
work = sys.argv[1]
D = [[float(x) for x in l.split()] for l in open("%s/v1.sadens.dat" % work) if l.strip()]
n = len(D)
asym = max(abs(D[i][j]-D[j][i]) for i in range(n) for j in range(n))
print("%s V5  D_avg symmetric                max|D-D^T| = %.2e" % ("PASS" if asym < 1e-14 else "FAIL", asym))
PY

python3 "$(dirname "$0")/moment_check.py" "$work" "$mp" "$w" "$lam"
