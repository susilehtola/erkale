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

  // ERKALE local index of the function with signed m in a shell of
  // angular momentum l. Spherical shells store m = -l..+l. ERKALE's
  // OptLM default keeps s and p as *cartesian*, but for l<2 those are
  // the same functions as the solid harmonics, in the order:
  //   l=0: [s]              (m=0 -> 0)
  //   l=1: [x, y, z]        (m=+1 -> 0, m=-1 -> 1, m=0 -> 2)
  // Cartesian d and higher are genuinely different (6d != 5d) and not
  // supported for a spherical TREXIO export.
  size_t erkale_local_index(const GaussianShell & sh, int m) {
    const int l = sh.get_am();
    if(sh.lm_in_use())
      return (size_t)(m + l);
    if(l == 0)
      return 0;
    if(l == 1) {
      if(m == 1)  return 0;   // x
      if(m == -1) return 1;   // y
      return 2;               // z (m == 0)
    }
    throw std::runtime_error("TREXIO spherical export needs spherical d and higher shells (run with UseLM true; cartesian d+ unsupported).");
  }

  // Per-shell permutation to TREXIO's storage order (m = 0,+1,-1,...).
  // perm[trexio_ao] = erkale_ao, so a quantity in ERKALE AO order is
  // read as q_erkale[perm[a]] at TREXIO position a. Shell order is
  // shared, and every shell has 2l+1 functions, so the global offset
  // is the same on both sides.
  std::vector<size_t> erkale_to_trexio_perm(const BasisSet & basis) {
    const std::vector<GaussianShell> & shells = basis.get_shells();
    std::vector<size_t> perm(basis.get_Nbf());
    for(size_t ish=0; ish<shells.size(); ish++) {
      const int l    = shells[ish].get_am();
      const size_t f = shells[ish].get_first_ind();
      for(int j=0; j<2*l+1; j++)
        perm[f + j] = f + erkale_local_index(shells[ish], trexio_sphe_m(j));
    }
    return perm;
  }
}

void chk_to_trexio(const std::string & chkfile, const std::string & trexiofile, bool verbose) {
  Checkpoint chk(chkfile, false);

  BasisSet basis;
  chk.read(basis);
  const std::vector<GaussianShell> & shells = basis.get_shells();
  const std::vector<nucleus_t> nuclei = basis.get_nuclei();
  const size_t Nbf = basis.get_Nbf();
  const size_t Nsh = shells.size();
  const size_t Nnuc = nuclei.size();

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
      charge[i]      = nuclei[i].Z;
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
    // One TREXIO shell per ERKALE shell; primitives flattened with a
    // shell_index back-pointer. Coefficients are ERKALE's normalized
    // contraction coefficients for the bare exp(-z r^2) primitive, so we
    // set prim_factor = shell_factor = 1 and let the overlap self-check
    // confirm the functions match.
    size_t Nprim=0;
    for(size_t ish=0; ish<Nsh; ish++)
      Nprim += shells[ish].get_contr().size();

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
      const int l = shells[ish].get_am();
      nuc_index[ish] = (int32_t) shells[ish].get_center_ind();
      shell_am[ish]  = (int32_t) l;
      // get_contr_normalized() returns the coefficients for *normalized*
      // primitives (the basis-file contraction coefficients); TREXIO
      // overlaps bare primitives scaled by prim_factor, so prim_factor
      // is the spherical-Gaussian primitive normalization
      //   N(z,l) = (2/pi)^{3/4} 2^l z^{(2l+3)/4} / sqrt((2l-1)!!)
      // (ERKALE's own convention), and the contracted AO comes out
      // unit-normalized with shell_factor = 1.
      const double fac = pow(M_2_PI, 0.75) * pow(2.0, l) / std::sqrt(doublefact(2*l-1));
      const std::vector<contr_t> c = shells[ish].get_contr_normalized();
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

    // --- ao (spherical) ---
    TX(trexio_write_ao_cartesian(tf, 0));
    TX(trexio_write_ao_num(tf, (int32_t) Nbf));
    std::vector<int32_t> ao_shell(Nbf);
    for(size_t ish=0; ish<Nsh; ish++) {
      const size_t f = shells[ish].get_first_ind();
      for(size_t k=0; k<shells[ish].get_Nbf(); k++)
        ao_shell[f+k] = (int32_t) ish;          // shell order is shared, so f is the TREXIO offset too
    }
    TX(trexio_write_ao_shell(tf, ao_shell.data()));
    std::vector<double> ao_norm(Nbf, 1.0);
    TX(trexio_write_ao_normalization(tf, ao_norm.data()));

    // --- mo ---
    const std::vector<size_t> perm = erkale_to_trexio_perm(basis);
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
  {
    const std::vector<size_t> perm = erkale_to_trexio_perm(basis);
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
  int32_t cart=0, up=0, dn=0;
  arma::mat Cfull;
  std::vector<double> occ, energy;
  std::vector<int32_t> spin;

  try {
    int32_t nnuc=0, nsh=0, nprim=0, nao=0, nmo=0;
    TX(trexio_read_nucleus_num(tf, &nnuc));
    TX(trexio_read_basis_shell_num(tf, &nsh));
    TX(trexio_read_basis_prim_num(tf, &nprim));
    TX(trexio_read_ao_num(tf, &nao));
    TX(trexio_read_mo_num(tf, &nmo));
    TX(trexio_read_electron_up_num(tf, &up));
    TX(trexio_read_electron_dn_num(tf, &dn));
    TX(trexio_read_ao_cartesian(tf, &cart));
    if(cart)
      throw std::runtime_error("TREXIO import currently supports spherical AOs only (ao_cartesian = 0).");
    Nsh=nsh; Nbf=nao; Nmo=nmo;

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
    std::vector<double>  exponent(nprim), coefficient(nprim);
    TX(trexio_read_basis_nucleus_index(tf, nuc_index.data()));
    TX(trexio_read_basis_shell_ang_mom(tf, shell_am.data()));
    TX(trexio_read_basis_shell_index(tf, shell_index.data()));
    TX(trexio_read_basis_exponent(tf, exponent.data()));
    TX(trexio_read_basis_coefficient(tf, coefficient.data()));
    for(int32_t ish=0; ish<nsh; ish++) {
      std::vector<contr_t> c;
      for(int32_t p=0; p<nprim; p++)
        if(shell_index[p]==ish) {
          contr_t t; t.z=exponent[p]; t.c=coefficient[p]; c.push_back(t);
        }
      basis.add_shell(nuc_index[ish], shell_am[ish], true, c, false);
    }
    basis.finalize();

    // MOs: undo the AO permutation (TREXIO order -> ERKALE order).
    const std::vector<size_t> perm = erkale_to_trexio_perm(basis);
    std::vector<double> moc(Nmo*Nbf);
    TX(trexio_read_mo_coefficient(tf, moc.data()));
    occ.resize(Nmo); energy.resize(Nmo); spin.assign(Nmo,0);
    if(trexio_has_mo_occupation(tf)==TREXIO_SUCCESS) trexio_read_mo_occupation(tf, occ.data());
    if(trexio_has_mo_energy(tf)==TREXIO_SUCCESS)      trexio_read_mo_energy(tf, energy.data());
    if(trexio_has_mo_spin(tf)==TREXIO_SUCCESS)        trexio_read_mo_spin(tf, spin.data());
    Cfull.set_size(Nbf, Nmo);
    for(size_t imo=0; imo<Nmo; imo++)
      for(size_t a=0; a<Nbf; a++)
        Cfull(perm[a], imo) = moc[imo*Nbf + a];   // ERKALE row perm[a] <- TREXIO pos a
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
  auto col = [&](const std::vector<size_t> & idx) {
    arma::mat C(Cfull.n_rows, idx.size());
    arma::vec E(idx.size());
    for(size_t k=0;k<idx.size();k++){ C.col(k)=Cfull.col(idx[k]); E(k)=energy[idx[k]]; }
    return std::make_pair(C,E);
  };
  if(restr) {
    auto ce=col(ia);
    chk.write("C", ce.first); chk.write("E", ce.second);
  } else {
    auto a=col(ia), b=col(ib);
    chk.write("Ca", a.first); chk.write("Ea", a.second);
    chk.write("Cb", b.first); chk.write("Eb", b.second);
  }
  chk.write("Restricted", restr);

  if(verbose) {
    printf("Wrote %s: %i nuclei, %i shells, %i AOs, %i MOs (%s).\n",
           chkfile.c_str(), (int)basis.get_Nnuc(), (int)Nsh, (int)Nbf, (int)Nmo,
           restr ? "restricted" : "unrestricted");
    fflush(stdout);
  }
}
