#!/usr/bin/env python3
"""Analytic checks on the per-state protonic moments and state averaging of
erkale_hneoci.  Driven by tests/moment_tests.sh, which produces the .props.json
sidecars this script consumes.  See README_MOMENTS.md for the physics.

All references are exact expectation values of the three-dimensional isotropic
harmonic oscillator of mass m and frequency w, written with b = m w:

    <z^2>_n  = (2 n_z + 1) / (2 b)          <x^2+y^2> = (n_x + n_y + 1) / b
    <z^4>_n  = 3 (2 n_z^2 + 2 n_z + 1) / (4 b^2)
"""
import json
import sys

npass = nfail = 0


def chk(name, got, exp, tol):
    global npass, nfail
    d = abs(got - exp)
    ok = d <= tol
    print("%s %-34s got %.12e  exact %.12e  |d|=%.2e (tol %.0e)"
          % ("PASS" if ok else "FAIL", name, got, exp, d, tol))
    if ok:
        npass += 1
    else:
        nfail += 1


def load(work, tag):
    with open("%s/%s.props.json" % (work, tag)) as f:
        return json.load(f)


def traces(tag, d, tol=1e-12):
    """V4: the protonic 1-RDM of every root carries exactly one proton."""
    for r in d["roots"]:
        chk("V4 %s Tr[D S_p] root %i" % (tag, r["index"]), r["trace"], 1.0, tol)


def main():
    work, mp, w, lam = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    b = mp * w

    # -- V6: the multi-root machinery must not perturb the CI ----------------
    print("\n--- V6: regression, default vs. multi-root run ---")
    e_def = open("%s/reg_default.out" % work).read()
    e_multi = open("%s/reg_multi.out" % work).read()

    def ci_energy(txt):
        for line in txt.splitlines():
            if line.startswith("CI energy"):
                return line.split()[2]
        return None

    global npass, nfail
    a, c = ci_energy(e_def), ci_energy(e_multi)
    if a is not None and a == c:
        print("PASS V6 CI energy bit-identical      (%s)" % a)
        npass += 1
    else:
        print("FAIL V6 CI energy differs: default=%s multi-root=%s" % (a, c))
        nfail += 1

    # The multi-root block must agree with the legacy scalar print.
    dm = load(work, "reg_multi")
    chk("V6 root 0 == printed CI energy", dm["roots"][0]["E"], float(c), 1e-14)
    traces("reg", dm)

    # -- V1: ground-state moments -------------------------------------------
    print("\n--- V1: ground-state protonic moments (isotropic trap) ---")
    d1 = load(work, "v1")
    r0 = d1["roots"][0]
    chk("V1 <z^2>_0     = 1/(2b)", r0["z2"], 1.0 / (2 * b), 1e-12)
    chk("V1 <x^2+y^2>_0 = 1/b", r0["x2py2"], 1.0 / b, 1e-12)
    chk("V1 <z^4>_0     = 3/(4b^2)", r0["z4"], 3.0 / (4 * b * b), 1e-14)
    traces("v1", d1)

    # -- V3: gaps of the isotropic ladder ------------------------------------
    print("\n--- V3: excitation gaps (isotropic n=1 manifold at w) ---")
    for i in (1, 2, 3):
        chk("V3 gap root %i = w" % i, d1["roots"][i]["gap"], w, 1e-8)

    # -- V5c: cylindrical symmetry of a nondegenerate state ------------------
    # Within the degenerate n=1 manifold the individual CI vectors are basis
    # dependent, so <x^2> == <y^2> holds per state only for root 0; over the
    # manifold only the sum is invariant.
    print("\n--- V5c: <x^2> == <y^2> (nondegenerate state, and summed over pi) ---")
    chk("V5c <x^2>_0 - <y^2>_0", r0["x2"] - r0["y2"], 0.0, 1e-14)
    sx = sum(d1["roots"][i]["x2"] for i in (1, 2, 3))
    sy = sum(d1["roots"][i]["y2"] for i in (1, 2, 3))
    chk("V5c sum_manifold <x^2>-<y^2>", sx - sy, 0.0, 1e-14)

    # -- V5b: manifold average, degeneracy invariant -------------------------
    # Equal weights over the three n=1 roots. The average is invariant to how
    # LAPACK orients the degenerate eigenvectors, so this is exact.
    print("\n--- V5b: equal-weight average over the n=1 manifold ---")
    sa = d1["state_average"]
    chk("V5b <z^2>_avg     = 5/(6b)", sa["z2"], 5.0 / (6 * b), 1e-12)
    chk("V5b <x^2+y^2>_avg = 5/(3b)", sa["x2py2"], 5.0 / (3 * b), 1e-12)
    chk("V5b <z^4>_avg     = 7/(4b^2)", sa["z4"], 7.0 / (4 * b * b), 1e-14)
    chk("V5b E_avg         = E_1", sa["E"], d1["roots"][1]["E"], 1e-12)

    # -- V2: excited-state moments, sigma_z resolved by the cross term -------
    # lam_cross < 0 pushes p_z below the pi doublet without mixing the states
    # (parity keeps the trap block diagonal in {s, p_x, p_y, p_z}), so root 1
    # is exactly the sigma_z oscillator state.
    print("\n--- V2: first sigma excited state (nondegenerate via lam_cross) ---")
    d2 = load(work, "v2")
    pz = d2["roots"][1]
    chk("V2 <z^2>_sigma     = 3/(2b)", pz["z2"], 3.0 / (2 * b), 1e-12)
    chk("V2 <z^4>_sigma     = 15/(4b^2)", pz["z4"], 15.0 / (4 * b * b), 1e-13)
    chk("V2 <x^2+y^2>_sigma = 1/b", pz["x2py2"], 1.0 / b, 1e-12)
    # The splitting itself pins the sign and scale of the M202 + M022 assembly.
    chk("V2 sigma/pi splitting", d2["roots"][3]["E"] - d2["roots"][1]["E"],
        abs(0.5 * lam / (b * b)), 1e-12)
    traces("v2", d2)

    # -- V5a: all weight on root 0 -------------------------------------------
    print("\n--- V5a: weights (1,0,0,0) reproduce root 0 ---")
    da = load(work, "v5a")
    sa = da["state_average"]
    g0 = da["roots"][0]
    for key in ("E", "z2", "x2py2", "z4"):
        chk("V5a %s_avg == root0" % key, sa[key], g0[key], 1e-14)

    # -- V5d: equal weights give the arithmetic mean --------------------------
    print("\n--- V5d: equal weights give the arithmetic mean ---")
    de = load(work, "v5d")
    sa = de["state_average"]
    n = len(de["roots"])
    for key in ("E", "z2", "x2py2", "z4"):
        mean = sum(r[key] for r in de["roots"]) / n
        chk("V5d %s_avg == mean over %i roots" % (key, n), sa[key], mean, 1e-13)

    print("\nRESULT: %i passed, %i failed" % (npass, nfail))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
