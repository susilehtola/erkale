#!/bin/bash
#
# Analytic validation of the external proton trap in erkale_hneoci.
# See README_TRAP.md for the operator, keywords and the theory behind T0-T3.
#
# The trap is added to the one-body protonic core Hamiltonian H0p, so the
# printed "Proton root" values are the eigenvalues of the *trapped free proton*
# (the electron does not enter H0p). In a single, matched Gaussian the proton
# energy equals the exact expectation value, which makes T1/T3 analytic.
#
# Usage:  tests/trap_tests.sh [path-to-erkale_hneoci]
# Env:    ERKALE_LIBRARY (basis directory) — defaults to <repo>/basis
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
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
pass=0; fail=0

# root <outfile> <index>  -> the printed proton eigenvalue
root() { awk -v i="$2" '$1=="Proton"&&$2=="root"&&$3==i{print $4}' "$1"; }
cienergy() { awk '/CI energy/{print $3}' "$1"; }

# assert_close <name> <got> <expect> <tol>
assert_close() {
  python3 - "$@" <<'PY'
import sys
name,got,exp,tol=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4])
try: g=float(got)
except: print("FAIL %-26s (no value parsed)"%name); sys.exit(1)
ok=abs(g-exp)<=tol
print("%s %-26s got %.10e  expect %.10e  |d|=%.2e (tol %.0e)"%(("PASS" if ok else "FAIL"),name,g,exp,abs(g-exp),tol))
sys.exit(0 if ok else 1)
PY
}
tally() { if [ "$1" -eq 0 ]; then pass=$((pass+1)); else fail=$((fail+1)); fi; }

# Single s primitive matched to an isotropic trap of frequency w: zeta = mp*w/2.
matched_basis() { # <w> <file>
  python3 -c "print('H     0');print('S   1   1.00');print('   %.10f   1.00'%($mp*$1/2))" > "$2"
  echo "****" >> "$2"
}
# Even-tempered s..f protonic basis for the anisotropic-oscillator convergence test.
et_basis() { # <file>
  python3 - "$1" <<'PY'
import sys
a0,beta,N=0.5,2.2,12
tag=["S","P","D","F"]; out=["H     0"]
for am in range(4):
    for i in range(N):
        out+= ["%s   1   1.00"%tag[am], "   %18.10f   1.00"%(a0*beta**i)]
out.append("****")
open(sys.argv[1],"w").write("\n".join(out)+"\n")
PY
}

echo; echo "=============================== T0 : regression ==============================="
# Trap off vs. trap enabled but identically zero must be bit-for-bit identical.
cat > "$work/t0off.run"  <<EOF
Basis cc-pVDZ
NEOTrap false
EOF
cat > "$work/t0zero.run" <<EOF
Basis cc-pVDZ
NEOTrap true
TrapOmegaUnit Eh
TrapOmegaPar 0.0
TrapOmegaPerp 0.0
TrapG 0.0
EOF
"$HN" "$work/t0off.run"  > "$work/t0off.out"  2>&1
"$HN" "$work/t0zero.run" > "$work/t0zero.out" 2>&1
eoff=$(cienergy "$work/t0off.out"); ezero=$(cienergy "$work/t0zero.out")
if [ -n "$eoff" ] && [ "$eoff" = "$ezero" ]; then
  echo "PASS T0 CI-energy bit-identical  ($eoff)"; tally 0
else
  echo "FAIL T0 CI-energy differ  off=$eoff  zero=$ezero"; tally 1
fi

echo; echo "======================== T1 : isotropic harmonic (E0 = 3/2 w) ================="
w=0.01; matched_basis $w "$work/s.gbs"
cat > "$work/t1.run" <<EOF
Basis cc-pVDZ
ProtonBasis $work/s.gbs
NEOTrap true
TrapOmegaUnit Eh
TrapOmegaPar $w
TrapOmegaPerp $w
TrapG 0.0
EOF
"$HN" "$work/t1.run" > "$work/t1.out" 2>&1
assert_close "T1 E0=3/2 w" "$(root "$work/t1.out" 0)" "$(python3 -c "print(1.5*$w)")" 1e-8; tally $?

echo; echo "=================== T3 : quartic, exact single-Gaussian expectation ==========="
# Isotropic g: <z^4>+<(x^2+y^2)^2> = 3/(4 m^2 w^2)+2/(m^2 w^2) => dE = (11/4) g w
g=0.001
cat > "$work/t3g.run" <<EOF
Basis cc-pVDZ
ProtonBasis $work/s.gbs
NEOTrap true
TrapOmegaUnit Eh
TrapOmegaPar $w
TrapOmegaPerp $w
TrapG $g
EOF
"$HN" "$work/t3g.run" > "$work/t3g.out" 2>&1
assert_close "T3 isotropic quartic" "$(root "$work/t3g.out" 0)" \
  "$(python3 -c "print(1.5*$w + 2.75*$g*$w)")" 1e-8; tally $?
# Pure cross term: dE = lam_cross * <z^2(x^2+y^2)> = lam_cross / (2 m^2 w^2)
L=0.01
cat > "$work/t3c.run" <<EOF
Basis cc-pVDZ
ProtonBasis $work/s.gbs
NEOTrap true
TrapOmegaUnit Eh
TrapOmegaPar $w
TrapOmegaPerp $w
TrapG 0.0
TrapLambdaCross $L
EOF
"$HN" "$work/t3c.run" > "$work/t3c.out" 2>&1
assert_close "T3 cross quartic" "$(root "$work/t3c.out" 0)" \
  "$(python3 -c "mp=$mp;w=$w;print(1.5*w + $L/(2*mp*mp*w*w))")" 1e-8; tally $?

echo; echo "============= T2 : cylindrical harmonic (E0 -> w_perp + 1/2 w_par) ============="
wpar=0.02; wperp=0.01; et_basis "$work/et.gbs"
cat > "$work/t2.run" <<EOF
Basis cc-pVDZ
ProtonBasis $work/et.gbs
NEOTrap true
TrapOmegaUnit Eh
TrapOmegaPar $wpar
TrapOmegaPerp $wperp
TrapG 0.0
EOF
"$HN" "$work/t2.run" > "$work/t2.out" 2>&1
E0=$(root "$work/t2.out" 0); E1=$(root "$work/t2.out" 1); E2=$(root "$work/t2.out" 2)
# Ground state converges from above to w_perp + 1/2 w_par.
assert_close "T2 E0=w_perp+w_par/2" "$E0" "$(python3 -c "print($wperp+0.5*$wpar)")" 5e-4; tally $?
# First excitation is a doubly-degenerate pi mode at w_perp above E0.
assert_close "T2 E1-E0 = w_perp" "$(python3 -c "print($E1-($E0))")" "$wperp" 2e-3; tally $?
assert_close "T2 pi degeneracy E2=E1" "$E2" "$E1" 1e-7; tally $?

echo; echo "==============================================================================="
echo "RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
