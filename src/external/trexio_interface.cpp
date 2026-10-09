/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2026
 * Copyright (c) 2010-2026, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "trexio_interface.h"
#include "../checkpoint.h"
#include "../basis.h"
#include "../cintenv.h"
#include "../eriworker.h"
#include "../elements.h"
#include "../mathf.h"

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

extern "C" {
#include <trexio.h>
}

// libtrexio after 2.6.1 can flag each shell as cartesian or spherical
// (ao.cartesian_shell); its TREXIO_AMBIGUOUS_CARTESIAN error code comes
// with it.
#ifdef TREXIO_AMBIGUOUS_CARTESIAN
#define ERKALE_TREXIO_CARTESIAN_SHELL
#endif

namespace {
  // Abort with a descriptive message if a TREXIO call failed.
  void check(trexio_exit_code rc, const char * what) {
    if(rc != TREXIO_SUCCESS) {
      std::ostringstream oss;
      oss << "TREXIO error in " << what << ": " << trexio_string_of_error(rc) << "\n";
      throw std::runtime_error(oss.str());
    }
  }
#define TX(call) check((call), #call)

  // ERKALE local index of the function with signed m in a spherical
  // shell of angular momentum l. Spherical d and higher shells store
  // m = -l..+l. s and p functions are the same whether cartesian (the
  // OptLM default) or spherical, and follow libcint's order:
  //   l=0: [s]              (m=0 -> 0)
  //   l=1: [x, y, z]        (m=+1 -> 0, m=-1 -> 1, m=0 -> 2)
  size_t erkale_local_index(int l, int m) {
    if(l == 0)
      return 0;
    if(l == 1) {
      if(m == 1)  return 0;   // x
      if(m == -1) return 1;   // y
      return 2;               // z (m == 0)
    }
    return (size_t)(m + l);
  }

  /// Number of functions of a TREXIO shell of angular momentum l
  size_t trexio_nfunc(int l, bool cart) {
    return cart ? (l+1)*(l+2)/2 : 2*l+1;
  }

  // Signed m of the j-th spherical function in TREXIO storage order
  // (0, +1, -1, +2, -2, ...), j in [0, 2l]. Inlined so the core path
  // does not depend on the (non-upstream) trexio_sphe_m header helper,
  // which lets the interface build against stock TREXIO.
  int trexio_signed_m(int j) {
    if(j == 0) return 0;
    return (j & 1) ? (j + 1) / 2 : -(j / 2);
  }

  // One contraction of an ERKALE shell. TREXIO has no general
  // contractions, so a generally contracted shell is written as one
  // TREXIO shell per contraction. ERKALE orders the functions of a
  // generally contracted shell contraction-slowest, so contraction ic
  // occupies its own span from first and the segments keep ERKALE's
  // order.
  struct segment_t {
    /// The ERKALE shell
    const GaussianShell * sh;
    /// Contraction index within the shell
    size_t ic;
    /// Index of the segment's first basis function
    size_t first;
  };
  std::vector<segment_t> segments(const std::vector<GaussianShell> & shells) {
    std::vector<segment_t> seg;
    for(size_t ish=0; ish<shells.size(); ish++) {
      const size_t nfunc = shells[ish].Nbf()/shells[ish].Nctr();
      for(size_t ic=0; ic<shells[ish].Nctr(); ic++)
        seg.push_back({&shells[ish], ic, shells[ish].first_ind() + ic*nfunc});
    }
    return seg;
  }

  // Cartesian flags of the TREXIO shells on export. ERKALE's own flag
  // decides for d and higher shells. s and p functions are the same
  // either way: they follow the d+ shells when those agree, so that the
  // file needs only the global ao.cartesian, and are spherical otherwise.
  std::vector<int32_t> cartesian_flags(const std::vector<segment_t> & seg) {
    // Flag shared by the d+ shells: -1 none seen yet, 2 mixed
    int dcart=-1;
    for(const segment_t & sg : seg)
      if(sg.sh->am() >= 2) {
        const int c = !sg.sh->lm_in_use();
        dcart = (dcart == -1 || dcart == c) ? c : 2;
      }
    std::vector<int32_t> cart(seg.size());
    for(size_t k=0; k<seg.size(); k++)
      cart[k] = (seg[k].sh->am() >= 2) ? !seg[k].sh->lm_in_use() : (dcart == 1);
    return cart;
  }

  // Per-segment permutation to TREXIO's storage order: m = 0,+1,-1,...
  // in a spherical shell, and the alphabetical order of the cartesian
  // functions -- which is ERKALE's -- in a cartesian one. cart holds
  // the TREXIO flag of each segment. perm[trexio_ao] = erkale_ao, so a
  // quantity in ERKALE AO order is read as q_erkale[perm[a]] at TREXIO
  // position a. Segment order is shared, and a segment has as many
  // functions on both sides, so the global offset is the same too.
  std::vector<size_t> erkale_to_trexio_perm(const BasisSet & basis, const std::vector<int32_t> & cart) {
    const std::vector<GaussianShell> shells = basis.shells();
    const std::vector<segment_t> seg = segments(shells);
    std::vector<size_t> perm(basis.Nbf());
    for(size_t k=0; k<seg.size(); k++) {
      const int l = seg[k].sh->am();
      for(size_t j=0; j<trexio_nfunc(l, cart[k]); j++)
        perm[seg[k].first + j] = seg[k].first + (cart[k] ? j : erkale_local_index(l, trexio_signed_m(j)));
    }
    return perm;
  }

  /// An AO matrix in ERKALE order as a TREXIO array (TREXIO AO order,
  /// row major; the matrices written here are symmetric).
  std::vector<double> ao_array(const arma::mat & M, const std::vector<size_t> & perm) {
    const size_t N=perm.size();
    std::vector<double> v(N*N);
    for(size_t a=0; a<N; a++)
      for(size_t b=0; b<N; b++)
        v[a*N+b] = M(perm[a], perm[b]);
    return v;
  }

  /// An AO matrix transformed to the MO basis C (row major).
  std::vector<double> mo_array(const arma::mat & M, const arma::mat & C) {
    const arma::mat X(C.t()*M*C);
    const size_t N=X.n_rows;
    std::vector<double> v(N*N);
    for(size_t i=0; i<N; i++)
      for(size_t j=0; j<N; j++)
        v[i*N+j] = X(i,j);
    return v;
  }

  /**
   * Write the AO electron repulsion integrals. Each symmetry-unique
   * integral above thr is stored once, in TREXIO's physicists' notation
   * <ab|cd> = (ac|bd) and TREXIO AO order. The loop covers the shell
   * quartets with is>=js, ks>=ls and is>=ks, a superset of the canonical
   * ones (for is==ks, which pair is larger depends on the functions, not
   * just on the shells), and the canonical integrals are picked out by
   * their TREXIO indices.
   */
  size_t write_ao_eri(trexio_t * tf, const BasisSet & basis, const std::vector<size_t> & perm, double thr) {
    // TREXIO index of each ERKALE AO
    std::vector<size_t> iperm(perm.size());
    for(size_t a=0; a<perm.size(); a++)
      iperm[perm[a]] = a;

    const std::vector<GaussianShell> shells = basis.shells();
    const size_t Nsh = shells.size();
    CintEnv cenv(basis);
    auto eri = make_eri_worker(cenv, 0.0, 1.0, 0.0);

    // Buffered sparse write
    const size_t bufsize = 1<<20;
    std::vector<int32_t> idx;
    std::vector<double> val;
    idx.reserve(4*bufsize);
    val.reserve(bufsize);
    int64_t offset = 0;
    auto flush = [&]() {
      if(val.empty())
        return;
      TX(trexio_write_ao_2e_int_eri(tf, offset, (int64_t) val.size(), idx.data(), val.data()));
      offset += val.size();
      idx.clear();
      val.clear();
    };

    // Pair index of a >= b
    auto pair = [](size_t a, size_t b) { return a*(a+1)/2 + b; };

    for(size_t is=0; is<Nsh; is++)
      for(size_t js=0; js<=is; js++)
        for(size_t ks=0; ks<=is; ks++)
          for(size_t ls=0; ls<=ks; ls++) {
            eri->compute(is, js, ks, ls);
            const std::vector<double> & ints = *eri->getp();
            const size_t Ni=shells[is].Nbf(), Nj=shells[js].Nbf(), Nk=shells[ks].Nbf(), Nl=shells[ls].Nbf();
            const size_t i0=shells[is].first_ind(), j0=shells[js].first_ind(), k0=shells[ks].first_ind(), l0=shells[ls].first_ind();
            for(size_t ii=0; ii<Ni; ii++)
              for(size_t jj=0; jj<Nj; jj++)
                for(size_t kk=0; kk<Nk; kk++)
                  for(size_t ll=0; ll<Nl; ll++) {
                    const double v = ints[((ii*Nj+jj)*Nk+kk)*Nl+ll];
                    if(std::abs(v) <= thr)
                      continue;
                    // Chemists' (ij|kl) in TREXIO indices
                    const size_t i=iperm[i0+ii], j=iperm[j0+jj], k=iperm[k0+kk], l=iperm[l0+ll];
                    if(i<j || k<l || pair(i,j)<pair(k,l))
                      continue;
                    // Physicists' <ik|jl>
                    const int32_t q[4] = {(int32_t) i, (int32_t) k, (int32_t) j, (int32_t) l};
                    idx.insert(idx.end(), q, q+4);
                    val.push_back(v);
                    if(val.size() == bufsize)
                      flush();
                  }
          }
    flush();
    return (size_t) offset;
  }
}

void chk_to_trexio(const std::string & chkfile, const std::string & trexiofile, bool eri, bool verbose) {
  Checkpoint chk(chkfile, false);

  BasisSet basis;
  chk.read(basis);
  const std::vector<GaussianShell> & shells = basis.shells();
  const std::vector<nucleus_t> nuclei = basis.nuclei();
  const size_t Nbf = basis.Nbf();
  // TREXIO shells: one per contraction of each ERKALE shell
  const std::vector<segment_t> seg = segments(shells);
  const size_t Nsh = seg.size();
  const size_t Nnuc = nuclei.size();
  // Cartesian flags of the TREXIO shells; one for all of them if they
  // agree, or one per shell (ao.cartesian_shell)
  const std::vector<int32_t> cart = cartesian_flags(seg);
  const bool cart_uniform = std::all_of(cart.begin(), cart.end(), [&](int32_t c) { return c == cart[0]; });
#ifndef ERKALE_TREXIO_CARTESIAN_SHELL
  if(!cart_uniform)
    throw std::runtime_error("The basis mixes cartesian and spherical shells, which needs a libtrexio with ao.cartesian_shell (newer than 2.6.1).");
#endif

  // Spin handling: "C" present -> restricted, else "Ca"/"Cb".
  const bool restr = chk.exist("C");
  arma::mat Ca, Cb;
  arma::vec Ea, Eb;
  int Nela=0, Nelb=0;
  chk.read("Nel-a", Nela);
  chk.read("Nel-b", Nelb);
  if(restr) {
    chk.read("C", Ca);
    chk.read("E", Ea);
  } else {
    chk.read("Ca", Ca);
    chk.read("Cb", Cb);
    chk.read("Ea", Ea);
    chk.read("Eb", Eb);
  }

  // Overwrite any pre-existing file (TREXIO refuses to open 'w' onto one).
  std::remove(trexiofile.c_str());
  trexio_exit_code rc;
  trexio_t * tf = trexio_open(trexiofile.c_str(), 'w', TREXIO_HDF5, &rc);
  if(tf == NULL)
    check(rc, "trexio_open (write)");

  try {
    // --- metadata ---
    TX(trexio_write_metadata_code_num(tf, 1));
    const char * codes[1] = {"ERKALE"};
    TX(trexio_write_metadata_code(tf, codes, 16));

    // --- nucleus ---
    TX(trexio_write_nucleus_num(tf, (int32_t) Nnuc));
    std::vector<double> charge(Nnuc), coord(3*Nnuc);
    std::vector<const char *> label(Nnuc);
    std::vector<std::string> labelstr(Nnuc);
    for(size_t i=0; i<Nnuc; i++) {
      charge[i]      = nuclei[i].bsse ? 0.0 : nuclei[i].Z;   // ghost atoms carry no charge
      coord[3*i+0]   = nuclei[i].r.x;   // ERKALE stores coordinates in bohr
      coord[3*i+1]   = nuclei[i].r.y;
      coord[3*i+2]   = nuclei[i].r.z;
      labelstr[i]    = nuclei[i].symbol;
      label[i]       = labelstr[i].c_str();
    }
    TX(trexio_write_nucleus_charge(tf, charge.data()));
    TX(trexio_write_nucleus_coord(tf, coord.data()));
    TX(trexio_write_nucleus_label(tf, label.data(), 32));
    // Nuclear repulsion (point charges; skip ghost/zero-charge centres).
    double erep=0.0;
    for(size_t i=0; i<Nnuc; i++)
      for(size_t j=0; j<i; j++) {
        const double dx=nuclei[i].r.x-nuclei[j].r.x;
        const double dy=nuclei[i].r.y-nuclei[j].r.y;
        const double dz=nuclei[i].r.z-nuclei[j].r.z;
        const double r=std::sqrt(dx*dx+dy*dy+dz*dz);
        if(r>0.0) erep += nuclei[i].Z*nuclei[j].Z/r;
      }
    TX(trexio_write_nucleus_repulsion(tf, erep));

    // --- electron ---
    TX(trexio_write_electron_up_num(tf, (int32_t) Nela));
    TX(trexio_write_electron_dn_num(tf, (int32_t) Nelb));

    // --- basis (Gaussian) ---
    // One TREXIO shell per ERKALE contraction; primitives flattened with a
    // shell_index back-pointer. See below for the coefficient and
    // prim_factor convention; the overlap self-check (when available)
    // confirms the functions match.
    size_t Nprim=0;
    for(size_t ish=0; ish<Nsh; ish++)
      Nprim += seg[ish].sh->contr(seg[ish].ic).size();

    TX(trexio_write_basis_type(tf, "Gaussian", 16));
    TX(trexio_write_basis_shell_num(tf, (int32_t) Nsh));
    TX(trexio_write_basis_prim_num(tf, (int32_t) Nprim));

    std::vector<int32_t> nuc_index(Nsh), shell_am(Nsh);
    std::vector<double>  shell_factor(Nsh, 1.0);
    std::vector<int32_t> r_power(Nsh, 0);   // standard Gaussians: r^l, no extra r power
    std::vector<int32_t> shell_index(Nprim);
    std::vector<double>  exponent(Nprim), coefficient(Nprim), prim_factor(Nprim);
    size_t ip=0;
    for(size_t ish=0; ish<Nsh; ish++) {
      const int l = seg[ish].sh->am();
      nuc_index[ish] = (int32_t) seg[ish].sh->center_ind();
      shell_am[ish]  = (int32_t) l;
      // contr_normalized() returns the coefficients for *normalized*
      // primitives (the basis-file contraction coefficients); TREXIO
      // overlaps bare primitives scaled by prim_factor, so prim_factor
      // is the normalization of the x^l (or, the same, the spherical)
      // Gaussian primitive
      //   N(z,l) = (2/pi)^{3/4} 2^l z^{(2l+3)/4} / sqrt((2l-1)!!)
      // (ERKALE's own convention), and the contracted AO comes out
      // unit-normalized with shell_factor = 1.
      const double fac = pow(M_2_PI, 0.75) * pow(2.0, l) / std::sqrt(doublefact(2*l-1));
      const std::vector<contr_t> c = seg[ish].sh->contr_normalized(seg[ish].ic);
      for(size_t k=0; k<c.size(); k++) {
        shell_index[ip] = (int32_t) ish;
        exponent[ip]    = c[k].z;
        coefficient[ip] = c[k].c;
        prim_factor[ip] = fac * pow(c[k].z, l/2.0 + 0.75);
        ip++;
      }
    }
    TX(trexio_write_basis_nucleus_index(tf, nuc_index.data()));
    TX(trexio_write_basis_shell_ang_mom(tf, shell_am.data()));
    TX(trexio_write_basis_shell_factor(tf, shell_factor.data()));
    TX(trexio_write_basis_r_power(tf, r_power.data()));
    TX(trexio_write_basis_shell_index(tf, shell_index.data()));
    TX(trexio_write_basis_exponent(tf, exponent.data()));
    TX(trexio_write_basis_coefficient(tf, coefficient.data()));
    TX(trexio_write_basis_prim_factor(tf, prim_factor.data()));

    // --- ao ---
    if(cart_uniform)
      TX(trexio_write_ao_cartesian(tf, cart[0]));
#ifdef ERKALE_TREXIO_CARTESIAN_SHELL
    else
      TX(trexio_write_ao_cartesian_shell(tf, cart.data()));
#endif
    TX(trexio_write_ao_num(tf, (int32_t) Nbf));
    std::vector<int32_t> ao_shell(Nbf);
    for(size_t ish=0; ish<Nsh; ish++)
      for(size_t k=0; k<trexio_nfunc(seg[ish].sh->am(), cart[ish]); k++)
        ao_shell[seg[ish].first+k] = (int32_t) ish;   // segment order is shared, so first is the TREXIO offset too
    TX(trexio_write_ao_shell(tf, ao_shell.data()));
    // ERKALE normalizes each cartesian function (the GAMESS convention),
    // so the functions of a cartesian shell carry their norm relative to
    // x^l, which prim_factor normalizes
    std::vector<double> ao_norm(Nbf, 1.0);
    for(size_t ish=0; ish<Nsh; ish++)
      if(!seg[ish].sh->lm_in_use()) {
        const std::vector<shellf_t> & cf = seg[ish].sh->cart_ref();
        for(size_t k=0; k<cf.size(); k++)
          ao_norm[seg[ish].first+k] = cf[k].relnorm;
      }
    TX(trexio_write_ao_normalization(tf, ao_norm.data()));

    // --- mo ---
    const std::vector<size_t> perm = erkale_to_trexio_perm(basis, cart);
    const size_t nmo_a = Ca.n_cols;
    const size_t nmo_b = restr ? 0 : Cb.n_cols;
    const size_t Nmo = nmo_a + nmo_b;
    TX(trexio_write_mo_type(tf, "Canonical", 16));
    TX(trexio_write_mo_num(tf, (int32_t) Nmo));

    // mo_coefficient is stored [imo][iao] (row-major), AO rows permuted
    // into TREXIO order.
    std::vector<double>  mocoef(Nmo*Nbf);
    std::vector<double>  occ(Nmo, 0.0), energy(Nmo, 0.0);
    std::vector<int32_t> spin(Nmo, 0);
    for(size_t imo=0; imo<nmo_a; imo++) {
      for(size_t a=0; a<Nbf; a++)
        mocoef[imo*Nbf + a] = Ca(perm[a], imo);
      energy[imo] = Ea(imo);
      occ[imo]    = restr ? ((imo<(size_t)Nela)?2.0:0.0) : ((imo<(size_t)Nela)?1.0:0.0);
      spin[imo]   = 0;
    }
    for(size_t imo=0; imo<nmo_b; imo++) {
      const size_t m = nmo_a + imo;
      for(size_t a=0; a<Nbf; a++)
        mocoef[m*Nbf + a] = Cb(perm[a], imo);
      energy[m] = Eb(imo);
      occ[m]    = (imo<(size_t)Nelb)?1.0:0.0;
      spin[m]   = 1;
    }
    TX(trexio_write_mo_coefficient(tf, mocoef.data()));
    TX(trexio_write_mo_occupation(tf, occ.data()));
    TX(trexio_write_mo_energy(tf, energy.data()));
    TX(trexio_write_mo_spin(tf, spin.data()));

    // --- one-electron integrals, in the AO and the MO basis ---
    // TREXIO's dipole operator is -r about the origin.
    const arma::mat Cmo = restr ? Ca : arma::mat(arma::join_rows(Ca, Cb));
    const arma::mat S(basis.overlap()), T(basis.kinetic()), V(basis.nuclear());
    const arma::mat H(T+V);
    const std::vector<arma::mat> r = basis.moment(1);
    const arma::mat dx(-r[0]), dy(-r[1]), dz(-r[2]);
    TX(trexio_write_ao_1e_int_overlap(tf, ao_array(S, perm).data()));
    TX(trexio_write_ao_1e_int_kinetic(tf, ao_array(T, perm).data()));
    TX(trexio_write_ao_1e_int_potential_n_e(tf, ao_array(V, perm).data()));
    TX(trexio_write_ao_1e_int_core_hamiltonian(tf, ao_array(H, perm).data()));
    TX(trexio_write_ao_1e_int_dipole_x(tf, ao_array(dx, perm).data()));
    TX(trexio_write_ao_1e_int_dipole_y(tf, ao_array(dy, perm).data()));
    TX(trexio_write_ao_1e_int_dipole_z(tf, ao_array(dz, perm).data()));
    TX(trexio_write_mo_1e_int_overlap(tf, mo_array(S, Cmo).data()));
    TX(trexio_write_mo_1e_int_kinetic(tf, mo_array(T, Cmo).data()));
    TX(trexio_write_mo_1e_int_potential_n_e(tf, mo_array(V, Cmo).data()));
    TX(trexio_write_mo_1e_int_core_hamiltonian(tf, mo_array(H, Cmo).data()));
    TX(trexio_write_mo_1e_int_dipole_x(tf, mo_array(dx, Cmo).data()));
    TX(trexio_write_mo_1e_int_dipole_y(tf, mo_array(dy, Cmo).data()));
    TX(trexio_write_mo_1e_int_dipole_z(tf, mo_array(dz, Cmo).data()));

    // --- two-electron integrals (optional: the list grows as N^4) ---
    if(eri) {
      const size_t nint = write_ao_eri(tf, basis, perm, 1e-14);
      if(verbose) {
        printf("Wrote %zu unique AO electron repulsion integrals.\n", nint);
        fflush(stdout);
      }
    }
  } catch(...) {
    trexio_close(tf);
    throw;
  }
  TX(trexio_close(tf));

  // Self-check: reopen read-only (HDF5 won't read back un-flushed
  // write-mode datasets) and compare TREXIO's computed AO overlap --
  // built from the basis as TREXIO interprets it -- against ERKALE's,
  // reordered into TREXIO AO order. Catches any normalization or
  // ordering mismatch in the basis/MO export.
  //
  // trexio_compute_ao_overlap / trexio_check_mo_orthonormality are not
  // part of upstream TREXIO; the build defines ERKALE_TREXIO_OVERLAP_HELPERS
  // only when the linked libtrexio provides them. Without them the export
  // still works and the round-trip test validates correctness.
#ifdef ERKALE_TREXIO_OVERLAP_HELPERS
  {
    const std::vector<size_t> perm = erkale_to_trexio_perm(basis, cart);
    trexio_exit_code orc;
    trexio_t * rf = trexio_open(trexiofile.c_str(), 'r', TREXIO_HDF5, &orc);
    if(rf != NULL) {
      // (a) AO overlap: TREXIO's basis vs ERKALE's (catches basis/
      //     normalization/ordering errors).
      std::vector<double> Strex(Nbf*Nbf);
      orc = trexio_compute_ao_overlap(rf, Strex.data());
      if(orc == TREXIO_SUCCESS) {
        const arma::mat Serk = basis.overlap();
        double maxerr=0.0;
        for(size_t a=0; a<Nbf; a++)
          for(size_t b=0; b<Nbf; b++)
            maxerr = std::max(maxerr, std::abs(Strex[a*Nbf+b] - Serk(perm[a], perm[b])));
        if(verbose) {
          printf("AO overlap self-check (TREXIO vs ERKALE): max abs deviation %.3e.\n", maxerr);
          fflush(stdout);
        }
        if(maxerr > 1e-8)
          fprintf(stderr, "Warning - TREXIO/ERKALE AO overlap differ by %.3e; basis normalization or ordering may be off.\n", maxerr);
      } else if(verbose) {
        printf("AO overlap self-check unavailable: %s.\n", trexio_string_of_error(orc));
        fflush(stdout);
      }
      // (b) MO orthonormality C^T S C = I (catches MO-coefficient
      //     ordering errors). Only meaningful for a single orthonormal
      //     set: in the unrestricted case the file holds alpha and beta
      //     MOs stacked, which are not mutually orthogonal across spin,
      //     so the combined-set check would spuriously fail -- skip it
      //     there (the round-trip validates the unrestricted MOs).
      if(restr) {
        double modev=0.0;
        trexio_exit_code mrc = trexio_check_mo_orthonormality(rf, &modev);
        if(mrc == TREXIO_SUCCESS) {
          if(verbose) {
            printf("MO orthonormality self-check (C^T S C = I): max abs deviation %.3e.\n", modev);
            fflush(stdout);
          }
          if(modev > 1e-6)
            fprintf(stderr, "Warning - exported MOs deviate from orthonormality by %.3e.\n", modev);
        }
      }
      trexio_close(rf);
    }
  }
#endif // ERKALE_TREXIO_OVERLAP_HELPERS

  if(verbose) {
    printf("Wrote %s: %i nuclei, %i shells, %i AOs.\n",
           trexiofile.c_str(), (int)Nnuc, (int)Nsh, (int)Nbf);
    fflush(stdout);
  }
}

void trexio_to_chk(const std::string & trexiofile, const std::string & chkfile, bool verbose) {
  trexio_exit_code rc;
  trexio_t * tf = trexio_open(trexiofile.c_str(), 'r', TREXIO_AUTO, &rc);
  if(tf == NULL)
    check(rc, "trexio_open (read)");

  BasisSet basis;
  size_t Nbf=0, Nmo=0, Nsh=0;
  int32_t up=0, dn=0;
  arma::mat Cfull;
  std::vector<double> occ, energy;
  std::vector<int32_t> spin;
  bool hasocc=false;

  try {
    int32_t nnuc=0, nsh=0, nprim=0, nao=0, nmo=0;
    TX(trexio_read_nucleus_num(tf, &nnuc));
    TX(trexio_read_basis_shell_num(tf, &nsh));
    TX(trexio_read_basis_prim_num(tf, &nprim));
    TX(trexio_read_ao_num(tf, &nao));
    TX(trexio_read_mo_num(tf, &nmo));
    TX(trexio_read_electron_up_num(tf, &up));
    TX(trexio_read_electron_dn_num(tf, &dn));
    Nsh=nsh; Nbf=nao; Nmo=nmo;

    // Cartesian flags of the shells: one for all of them (ao.cartesian)
    // or one per shell (ao.cartesian_shell). A file has one or the other.
    std::vector<int32_t> cart(nsh);
    if(trexio_has_ao_cartesian(tf)==TREXIO_SUCCESS) {
      int32_t c;
      TX(trexio_read_ao_cartesian(tf, &c));
      cart.assign(nsh, c);
    }
#ifdef ERKALE_TREXIO_CARTESIAN_SHELL
    else if(trexio_has_ao_cartesian_shell(tf)==TREXIO_SUCCESS)
      TX(trexio_read_ao_cartesian_shell(tf, cart.data()));
#endif
    else
      throw std::runtime_error("The TREXIO file has no ao.cartesian. If it has ao.cartesian_shell instead, ERKALE needs to be built with a libtrexio newer than 2.6.1.");

    // nuclei
    std::vector<double> charge(nnuc), coord(3*nnuc);
    TX(trexio_read_nucleus_charge(tf, charge.data()));
    TX(trexio_read_nucleus_coord(tf, coord.data()));
    for(int32_t i=0; i<nnuc; i++) {
      nucleus_t nuc;
      nuc.ind=i;
      nuc.r.x=coord[3*i+0]; nuc.r.y=coord[3*i+1]; nuc.r.z=coord[3*i+2];
      nuc.Z=(int) std::round(charge[i]);
      nuc.Q=0;
      nuc.bsse=false;
      nuc.symbol=element_symbols[nuc.Z];
      basis.add_nucleus(nuc);
    }

    // basis -> shells
    std::vector<int32_t> nuc_index(nsh), shell_am(nsh), shell_index(nprim);
    std::vector<double>  exponent(nprim), coefficient(nprim), prim_factor(nprim);
    std::vector<double>  shell_factor(nsh, 1.0), ao_norm(nao, 1.0);
    TX(trexio_read_basis_nucleus_index(tf, nuc_index.data()));
    TX(trexio_read_basis_shell_ang_mom(tf, shell_am.data()));
    TX(trexio_read_basis_shell_index(tf, shell_index.data()));
    TX(trexio_read_basis_exponent(tf, exponent.data()));
    TX(trexio_read_basis_coefficient(tf, coefficient.data()));
    TX(trexio_read_basis_prim_factor(tf, prim_factor.data()));
    if(trexio_has_basis_shell_factor(tf)==TREXIO_SUCCESS)
      TX(trexio_read_basis_shell_factor(tf, shell_factor.data()));
    if(trexio_has_ao_normalization(tf)==TREXIO_SUCCESS)
      TX(trexio_read_ao_normalization(tf, ao_norm.data()));
    // Norm of the x^l (or, the same, the spherical) function of each
    // shell, which ERKALE's functions have normalized
    std::vector<double> shell_norm(nsh);
    for(int32_t ish=0; ish<nsh; ish++) {
      // TREXIO scales the bare primitive exp(-z r^2) by prim_factor, so
      // coefficient*prim_factor is the coefficient of the bare primitive,
      // which is what ERKALE stores; finalize then normalizes the
      // contraction.
      std::vector<contr_t> c;
      for(int32_t p=0; p<nprim; p++)
        if(shell_index[p]==ish) {
          contr_t t; t.z=exponent[p]; t.c=coefficient[p]*prim_factor[p]; c.push_back(t);
        }
      const int l = shell_am[ish];
      double S=0.0;
      for(const contr_t & p : c)
        for(const contr_t & q : c)
          S += p.c*q.c/pow(p.z+q.z, l+1.5);
      shell_norm[ish] = shell_factor[ish]*std::sqrt(pow(M_PI,1.5)*doublefact(2*l-1)/pow(2.0,l)*S);
      // Spherical d and higher shells as such; s and p are cartesian as
      // in ERKALE's OptLM default, and the same functions either way.
      basis.add_shell(nuc_index[ish], l, l>=2 && !cart[ish], c, false);
    }
    // Contraction-slowest segments on the same center with the same
    // exponents are regrouped into generally contracted shells here.
    basis.finalize();

    // MOs: undo the AO permutation (TREXIO order -> ERKALE order).
    const std::vector<size_t> perm = erkale_to_trexio_perm(basis, cart);
    // TREXIO AO a is ao_scale[a] times ERKALE's normalized function
    // perm[a]: its ao.normalization times its shell's norm, over the
    // norm of a cartesian function relative to x^l
    std::vector<double> ao_scale(Nbf);
    {
      const std::vector<GaussianShell> shells = basis.shells();
      const std::vector<segment_t> seg = segments(shells);
      for(size_t k=0; k<seg.size(); k++)
        for(size_t j=0; j<trexio_nfunc(seg[k].sh->am(), cart[k]); j++) {
          const size_t a = seg[k].first + j;
          ao_scale[a] = ao_norm[a]*shell_norm[k];
          if(cart[k])
            ao_scale[a] /= seg[k].sh->cart_ref()[j].relnorm;
        }
    }
    std::vector<double> moc(Nmo*Nbf);
    TX(trexio_read_mo_coefficient(tf, moc.data()));
    occ.resize(Nmo); energy.resize(Nmo); spin.assign(Nmo,0);
    hasocc = (trexio_has_mo_occupation(tf)==TREXIO_SUCCESS);
    if(hasocc) trexio_read_mo_occupation(tf, occ.data());
    if(trexio_has_mo_energy(tf)==TREXIO_SUCCESS)      trexio_read_mo_energy(tf, energy.data());
    if(trexio_has_mo_spin(tf)==TREXIO_SUCCESS)        trexio_read_mo_spin(tf, spin.data());
    Cfull.set_size(Nbf, Nmo);
    for(size_t imo=0; imo<Nmo; imo++)
      for(size_t a=0; a<Nbf; a++)
        Cfull(perm[a], imo) = ao_scale[a]*moc[imo*Nbf + a];   // ERKALE row perm[a] <- TREXIO pos a
  } catch(...) {
    trexio_close(tf);
    throw;
  }
  TX(trexio_close(tf));

  // Split into alpha/beta blocks by spin and write the checkpoint.
  std::vector<size_t> ia, ib;
  for(size_t i=0; i<Nmo; i++) (spin[i]==0 ? ia : ib).push_back(i);
  const bool restr = ib.empty();

  Checkpoint chk(chkfile, true);
  chk.write(basis);
  chk.write("Nel-a", up);
  chk.write("Nel-b", dn);
  chk.write("Nel", (int)(up+dn));
  // Orbitals, energies and occupations of one spin block. Without
  // mo_occupation in the file, occupy the lowest orbitals (aufbau; in a
  // restricted file the first dn orbitals doubly, the next up-dn singly).
  struct block_t { arma::mat C; arma::vec E, occ; };
  auto col = [&](const std::vector<size_t> & idx, int nel, int nel2) {
    block_t b;
    b.C.set_size(Cfull.n_rows, idx.size());
    b.E.set_size(idx.size());
    b.occ.set_size(idx.size());
    for(size_t k=0;k<idx.size();k++) {
      b.C.col(k)=Cfull.col(idx[k]);
      b.E(k)=energy[idx[k]];
      if(hasocc)
        b.occ(k)=occ[idx[k]];
      else
        b.occ(k)=((int) k<nel2 ? 1.0 : 0.0) + ((int) k<nel ? 1.0 : 0.0);
    }
    return b;
  };
  // Density matrix of a spin block
  auto density = [](const block_t & b) -> arma::mat {
    return b.C*arma::diagmat(b.occ)*b.C.t();
  };
  if(restr) {
    const block_t r=col(ia, up, dn);
    chk.write("C", r.C); chk.write("E", r.E);
    chk.write("P", density(r));
  } else {
    const block_t a=col(ia, up, 0), b=col(ib, dn, 0);
    chk.write("Ca", a.C); chk.write("Ea", a.E);
    chk.write("Cb", b.C); chk.write("Eb", b.E);
    const arma::mat Pa=density(a), Pb=density(b);
    chk.write("Pa", Pa); chk.write("Pb", Pb); chk.write("P", Pa+Pb);
  }
  chk.write("Restricted", restr);

  if(verbose) {
    printf("Wrote %s: %i nuclei, %i shells, %i AOs, %i MOs (%s).\n",
           chkfile.c_str(), (int)basis.Nnuc(), (int)Nsh, (int)Nbf, (int)Nmo,
           restr ? "restricted" : "unrestricted");
    fflush(stdout);
  }
}
