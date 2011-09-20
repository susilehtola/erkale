/*
 * This file is written by Arto Sakko and Jussi Lehtola, 2011
 * Copyright (c) 2011, Arto Sakko and Jussi Lehtola
 *
 */

/*
 *
 *
 *                   This file is part of
 *
 *                     E  R  K  A  L  E
 *                            -
 *                       DFT from Hel
 *
 * ERKALE is written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

/**
 * These two papers are often quoted in the commens of casida.h and casida.cpp:
 *
 * M. E. Casida, C. Jamorski, F. Bohr, J. Guan, and D.R. Salahub,
 * "Time-dependent density-functional response theory for molecules"
 * in Theoretical and Computational Modeling of NLO and Electronic Materials,
 * edited by S.P. Karna and A.T. Yeates (ACS Press: Washington, D.C., 1996)
 * (https://sites.google.com/site/markcasida/publications)
 *
 * C. Jamorski, M. E. Casida, and D. R. Salahub, "Dynamic polarizabilities
 * and excitation spectra from a molecular implementation of time-dependent
 * density-functional response theory: N2 as a case study",
 * J. Chem. Phys. 104, pp. 5134-5147 (1996).
 */

#ifndef ERKALE_CASIDA
#define ERKALE_CASIDA

#include "basis.h"
#include "density_fitting.h"
#include "dftgrid.h"
#include "linalg.h"
#include "scf.h"
#include "settings.h"

#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

#include <xc.h>

/// What kind of coupling is used
enum coupling_mode {
  /// Independent Particles Approximation: simple transitions between the ground state KS-orbitals, no coupling between electron-hole pairs.
  IPA,
  /// Random Phase Approximation (i.e. time-dependent Hartree approximation): dynamic Coulomb coupling between pairs of single-particle transitions.
  RPA,
  /// Time-Dependent Local Density Approximation: Coulomb and exchange-correlation coupling between pairs of single-particle transitions.
  TDLDA
};

/**
 * Pair of initial and final KS-orbitals, between which a single-particle
 * excitation takes place. Both states have the same spin.
 */
typedef struct {
  /// Initial state
  int i;
  /// Final state
  int f;
} states_pair_t;

/// Class for performing Casida calculations
class Casida {
  /// List of electron-hole pairs included in calculation
  std::vector< std::vector<states_pair_t> > pairs;
  /// How to couple electron-hole pairs (IPA, RPA, TDLDA)
  enum coupling_mode coupling;

  /// Occupancies of orbitals
  std::vector< arma::vec > f;
  /// Number of occupied orbitals
  std::vector<size_t> nocc;
  /// Number of virtual orbitals
  std::vector<size_t> nvirt;

  /// MO energies for spin up and spin down
  std::vector<arma::vec> E;
  /// MO coefficient matrices for spin up and spin down
  std::vector<arma::mat> C;
  /// Density matrices
  std::vector<arma::mat> P;
  /// Dipole matrix elements: [nspin][3][norb,norb]
  std::vector< std::vector<arma::mat> > dipmat;

  // K matrix
  arma::mat K;

  /// Eigenvalues of Casidas equation (and later on, the excitation energies)
  arma::vec w_i;
  /// Eigenvectors of Casidas equation
  arma::mat F_i;

  /**
   * This routine constructs the Coulomb coupling matrix
   * \lf$ K_{\rm Coul}(ij\sigma, kl\tau) = \int d^3r d^3r' \psi_{i\sigma}({\bf r}) \psi_{j\sigma}({\bf r}) \frac 1 {\left| {\bf r} - {\bf r}' \right|} \psi_{k\tau}(r') \psi_{l\tau}(r') \lf$
   * (Eq. 2.6 in Jamorski et al [1996]).
   *
   * Only the same-spin couplings are computed, since spin up - spin
   * down couplings are not necessary even for TDLDA.
   *
   * To save memory, the result is stored in the K matrix.
   */
  void Kcoul(const BasisSet & basis);

  /**
   * This routine constructs the exchange-correlation coupling matrix
   * \lf$ K_{\rm XC}(ij\sigma, kl\tau) = \int d^3r d^3r' \psi_{i\sigma}({\bf r}) \psi_{j\sigma}({\bf r}) \frac {\delta^2 E_{xc} [\rho_\uparrow,\rho_\downarrow]} {\delta \rho_\sigma ({\bf r}) \delta \rho_\tau ({\bf r})} \psi_{k\tau}(r') \psi_{l\tau}(r') \lf$
   *
   * Only the same-spin couplings are computed, since spin up - spin
   * down couplings are not necessary even for TDLDA.  
   *
   * To save memory, the result is stored in the K matrix.
   */
  void Kxc(const BasisSet & bas, double tol, int x_func, int c_func);

  /* Helper routines */

  /// Calculate \lf$ (\epsilon_{l\tau} - \epsilon_{k\tau})^2 \lf$
  double esq(states_pair_t ip, bool ispin) const;

  /// Calculate \lf$ \sqrt{ (f_{i\sigma} - f_{j\sigma}) (\epsilon_{j\sigma}-\epsilon_{i\sigma}) } \lf$
  double fe(states_pair_t ip, bool ispin) const;

  /// Form pairs and occupations
  void form_pairs(const Settings & set, const BasisSet & bas, size_t Norb, bool pol);

  /// Form dipole matrix
  void calc_dipole(const BasisSet & bas);

  /// Common routines for constructors
  void parse_args(const Settings & set, const BasisSet & basis, size_t Norbs);

  /// Construct the K matrices
  void calc_K(const Settings & set, const BasisSet & bas);

  /// Compute the Coulomb fitting integrals \lf$ (\mu \nu | a) \lf$ and the matrix \lf$ (a|b)^{-1} \lf$
  void coulomb_fit(const BasisSet & basis, std::vector<arma::mat> & munu, arma::mat & ab_inv) const;

 public:
  /// Constructor for spin-unpolarized calculation
  Casida(const Settings & set, const BasisSet & basis, const arma::vec & E, const arma::mat & C, const arma::mat & P);
  /// Constructor for spin-polarized calculation
  Casida(const Settings & set, const BasisSet & basis, const arma::vec & Ea, const arma::vec & Eb, const arma::mat & Ca, const arma::mat & Cb, const arma::mat & Pa, const arma::mat & Pb);
  /// Destructor
  ~Casida();

  /// Solve the Casida equation
  void solve();

  // Routines for calculating and outputting the actual spectra
  void absorption() const;
};



#endif
