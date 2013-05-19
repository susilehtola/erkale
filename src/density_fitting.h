/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2012
 * Copyright (c) 2010-2012, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


/**
 * \class DensityFit
 *
 * \brief Density fitting / RI routines
 *
 * This class contains density fitting and resolution of the identity
 * routines used for the approximate calculation of the Coulomb and
 * exchange operators J and K.
 *
 * The RI-JK implementation is based on the procedure described in
 *
 * F. Weigend, "A fully direct RI-HF algorithm: Implementation,
 * optimised auxiliary basis sets, demonstration of accuracy and
 * efficiency", Phys. Chem. Chem. Phys. 4, 4285 (2002).
 *
 *
 * If only RI-J is necessary, then the procedure described in
 *
 * K. Eichkorn, O. Treutler, H. Öhm, M. Häser and R. Ahlrichs,
 * "Auxiliary basis sets to approximate Coulomb potentials",
 * Chem. Phys. Lett. 240 (1995), 283-290.
 *
 * is used.
 *
 * \author Susi Lehtola
 * \date 2012/08/22 23:53
 */




#ifndef ERKALE_DENSITYFIT
#define ERKALE_DENSITYFIT

#include "global.h"
#include "basis.h"

/// Density fitting routines
class DensityFit {
  /// Amount of orbital basis functions
  size_t Nbf;
  /// Amount of auxiliary basis functions
  size_t Naux;
  /// Direct calculation? (Compute three-center integrals on-the-fly)
  bool direct;
  /// Hartree-Fock calculation?
  bool hf;

  /// Orbital shells
  std::vector<GaussianShell> orbshells;
  int maxorbam;
  size_t maxorbcontr;
  /// Density fitting shells
  std::vector<GaussianShell> auxshells;
  int maxauxam;
  size_t maxauxcontr;
  /// Dummy shell
  GaussianShell dummy;

  /// Index of dummy function
  size_t dummyind;

  /// List of unique orbital shell pairs
  std::vector<shellpair_t> orbpairs;

  /// Index helper
  std::vector<size_t> iidx;

  /// Screening matrix
  arma::mat screen;

  /// Integrals \f$ ( \alpha | \mu \nu) \f$
  std::vector<double> a_munu;
  /// Integrals \f$ ( \alpha | \mu \mu) \f$ (needed for xc fitting)
  std::vector<double> a_mu;

  /// \f$ ( \alpha | \beta) \f$
  arma::mat ab;
  /// \f$ ( \alpha | \beta)^-1 \f$
  arma::mat ab_inv;
  /// \f$ ( \alpha | \beta)^-1/2 \f$
  arma::mat ab_invh;

 public:
  /// Constructor
  DensityFit();
  /// Destructor
  ~DensityFit();

  /// Compute integrals, use given linear dependency treshhold
  void fill(const BasisSet & orbbas, const BasisSet & auxbas, bool direct, double threshold, bool hf=false);
  /// Compute index in integral table
  size_t idx(size_t ia, size_t imu, size_t inu) const;

  /// Compute estimate of necessary memory
  size_t memory_estimate(const BasisSet & orbbas, const BasisSet & auxbas, bool direct) const;

  /// Compute expansion coefficients c
  arma::vec compute_expansion(const arma::mat & P) const;
  /// Compute expansion coefficients c
  std::vector<arma::vec> compute_expansion(const std::vector<arma::mat> & P) const;

  /// Invert expansion (needed for XC fitting)
  arma::mat invert_expansion(const arma::vec & gamma) const;
  /// Invert only the diagonal of the expansion
  arma::vec invert_expansion_diag(const arma::vec & gamma) const;

  /// Get Coulomb matrix from P
  arma::mat calc_J(const arma::mat & P) const;
  /// Get Coulomb matrix from P
  std::vector<arma::mat> calc_J(const std::vector<arma::mat> & P) const;

  /// Get exchange matrix from orbitals with occupation numbers occs
  arma::mat calc_K(const arma::mat & C, const std::vector<double> & occs, size_t memlimit) const;

  /// Get the number of orbital functions
  size_t get_Norb() const;
  /// Get the number of auxiliary functions
  size_t get_Naux() const;
  /// Get the three-electron integral
  double get_a_munu(size_t ia, size_t imu, size_t inu) const;
  /// Get ab_inv
  arma::mat get_ab_inv() const;
};


#endif
