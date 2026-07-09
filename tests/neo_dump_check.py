#!/usr/bin/env python3
"""Validate a neo_dump.h5 the way a correlation consumer must read it.

This is the reference implementation of the contract in neo_dump_format.md
(v2): the dump stores the SCF orbitals and the core Hamiltonians, and the
consumer rebuilds its own reference Fock and semicanonicalizes.

Checks, per species:
  C1  orthonormality           C^T S C = I
  C2  occupation contract      occ = (occ_full,)*nocc + (0,)*(nmo-nocc)
  C3  density trace            Tr[D S] = particle count
  C4  reference energy         rebuilt from hcore + B + eri_ep equals e_scf
  C5  semicanonicalization     leaves the density invariant, and the
                               semicanonical F is block diagonal

C4 is the one that catches a wrong occupied subspace: if the dumped orbitals
do not span the SCF occupied space, the mean-field energy does not come back.

Usage:  tests/neo_dump_check.py <dump.h5> [<dump.h5> ...]
"""
import sys

import h5py
import numpy as np

npass = nfail = 0


def chk(name, ok, detail=""):
    global npass, nfail
    print("%s %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    if ok:
        npass += 1
    else:
        nfail += 1


def close(name, got, exp, tol):
    d = abs(got - exp)
    chk(name, d <= tol, "got %.12e  exp %.12e  |d|=%.2e (tol %.0e)" % (got, exp, d, tol))


def species(f, grp, suffix=""):
    """Read one species block. Occupation lives in the column order."""
    g = f[grp]
    C = g["C" + suffix][:]
    occ = g["occ" + suffix][:]
    nocc = int(g["nocc" + suffix][()])
    return dict(C=C, occ=occ, nocc=nocc, S=g["overlap"][:], H=g["hcore"][:],
                B=g["B"][:] if "B" in g else None, name=grp.strip("/") + suffix)


def density(r):
    """D = C diag(occ) C^T -- the SCF density, by construction of the dump."""
    return (r["C"] * r["occ"]) @ r["C"].T


def jk(B, D):
    """Coulomb and exchange from the engine factor B[P,mu,nu]."""
    J = np.einsum("Qmn,Qls,ls->mn", B, B, D, optimize=True)
    K = np.einsum("Qml,ls,Qsn->mn", B, D, B, optimize=True)
    return J, K


def semicanonicalize(C, F, nocc):
    """Diagonalize F within the occupied and the virtual block separately.
    Returns rotated C and eps. Leaves both subspaces -- hence D -- untouched."""
    Fmo = C.T @ F @ C
    Fmo = 0.5 * (Fmo + Fmo.T)
    nmo = C.shape[1]
    U = np.zeros((nmo, nmo))
    eps = np.zeros(nmo)
    for lo, hi in ((0, nocc), (nocc, nmo)):
        if hi <= lo:
            continue
        w, v = np.linalg.eigh(Fmo[lo:hi, lo:hi])
        eps[lo:hi] = w
        U[lo:hi, lo:hi] = v
    return C @ U, eps


def check_species(r, occ_full, nparticle):
    n = r["C"].shape[1]
    orth = np.abs(r["C"].T @ r["S"] @ r["C"] - np.eye(n)).max()
    chk("C1 %s orthonormal" % r["name"], orth < 1e-8, "||C'SC-I||=%.2e" % orth)

    want = np.array([occ_full] * r["nocc"] + [0.0] * (n - r["nocc"]))
    chk("C2 %s occupation contract" % r["name"], np.abs(r["occ"] - want).max() < 1e-10)

    D = density(r)
    close("C3 %s Tr[D S]" % r["name"], float(np.trace(D @ r["S"])), nparticle, 1e-10)
    return D


def coulomb_exchange(f, r, D):
    """(J,K) for one species, from the B factor or the dense rank-4 tensor."""
    if r["B"] is not None:
        return jk(r["B"], D)
    eri = f["/eri_ee" if r["name"].startswith("electron") else "/eri_pp"][:]
    return (np.einsum("mnls,ls->mn", eri, D, optimize=True),
            np.einsum("mlsn,ls->mn", eri, D, optimize=True))


def main():
    global npass, nfail
    for path in sys.argv[1:]:
        print("\n=== %s ===" % path)
        f = h5py.File(path, "r")
        restricted = bool(f.attrs["restricted_electrons"])
        nel = int(f.attrs["n_electrons"])
        npr = int(f.attrs["n_quantum_protons"])
        e_scf = float(f.attrs["e_scf"])
        e_cls = float(f.attrs["e_classical"])

        # electron blocks: one (RHF, occ 2) or two (UHF, occ 1 each)
        if restricted:
            eblk = [species(f, "/electron")]
            occ_full = [2.0]
        else:
            eblk = [species(f, "/electron", "_a"), species(f, "/electron", "_b")]
            occ_full = [1.0, 1.0]
        p = species(f, "/proton")

        Ds = [check_species(r, o, o * r["nocc"]) for r, o in zip(eblk, occ_full)]
        De = sum(Ds)
        close("C3 electron total Tr[D S]", float(np.trace(De @ eblk[0]["S"])), nel, 1e-10)
        Dp = check_species(p, 1.0, npr)

        eri_ep = f["/eri_ep"][:]
        Jp, Kp = coulomb_exchange(f, p, Dp)
        Je, _ = coulomb_exchange(f, eblk[0], De)          # Coulomb from the total density
        Ks = [coulomb_exchange(f, r, d)[1] for r, d in zip(eblk, Ds)]

        # cross-species Coulomb, bare and positive; the -1 sign is applied here
        Vpe = np.einsum("mnab,ab->mn", eri_ep, Dp, optimize=True)  # on electrons
        Vep = np.einsum("mnab,mn->ab", eri_ep, De, optimize=True)  # on protons

        # C4: the SCF mean-field energy, rebuilt entirely from dumped quantities.
        # c_x = 1/4 on the total density for RHF; 1/2 per spin for UHF and for
        # the high-spin protons.
        Eexch = (-0.25 * np.sum(De * Ks[0]) if restricted
                 else -0.5 * sum(np.sum(d * k) for d, k in zip(Ds, Ks)))
        E = (sum(np.sum(d * r["H"]) for d, r in zip(Ds, eblk)) + np.sum(Dp * p["H"])
             + 0.5 * np.sum(De * Je) + Eexch
             + 0.5 * np.sum(Dp * Jp) - 0.5 * np.sum(Dp * Kp)
             - np.sum(De * Vpe) + e_cls)
        close("C4 reference energy == e_scf", E, e_scf, 1e-6)

        # C5: rebuild each reference Fock and semicanonicalize.
        # electron: the true SCF Fock. proton: self-interaction-free (no Jp/Kp).
        blocks = [(r, d, r["H"] + Je - (0.5 if restricted else 1.0) * k - Vpe)
                  for r, d, k in zip(eblk, Ds, Ks)]
        blocks.append((p, Dp, p["H"] - Vep))
        for r, D, F in blocks:
            Cs, eps = semicanonicalize(r["C"], F, r["nocc"])
            Ds_new = (Cs * r["occ"]) @ Cs.T
            chk("C5 %s semicanonical D invariant" % r["name"],
                np.abs(Ds_new - D).max() < 1e-10, "max|dD|=%.2e" % np.abs(Ds_new - D).max())
            Fmo = Cs.T @ F @ Cs
            no = r["nocc"]
            blk = max(np.abs(Fmo[:no, :no] - np.diag(np.diag(Fmo[:no, :no]))).max(),
                      np.abs(Fmo[no:, no:] - np.diag(np.diag(Fmo[no:, no:]))).max())
            ov = np.abs(Fmo[:no, no:]).max() if no < len(eps) else 0.0
            chk("C5 %s blocks diagonalized" % r["name"], blk < 1e-10,
                "max offdiag %.2e ; surviving occ-vir |F_ia| %.2e" % (blk, ov))
            print("       %-14s eps[occ] = %s" % (r["name"], np.array2string(eps[:no], precision=6)))
        f.close()

    print("\nRESULT: %i passed, %i failed" % (npass, nfail))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
