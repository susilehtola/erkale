/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2012
 * Copyright (c) 2010-2012, Jussi Lehtola
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
 * The implementation is based on the procedure described in
 *
 * F. Weigend, "A fully direct RI-HF algorithm: Implementation,
 * optimised auxiliary basis sets, demonstration of accuracy and
 * efficiency", Phys. Chem. Chem. Phys. 4, 4285 (2002).
 * 
 * \author Jussi Lehtola
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

  /// Basis set containing first orbital, then auxiliary and lastly dummy function
  BasisSet totbas;
  /// Indices of orbital shells
  std::vector<size_t> orbind;
  /// Indices of density fitting shells
  std::vector<size_t> auxind;
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
  /// \f$ ( \alpha | \beta)^-1 \f$
  arma::mat ab_inv;
  /// \f$ ( \alpha | \beta)^-1/2 \f$
  arma::mat ab_invh;

 public:
  /// Constructor
  DensityFit();	 
  /// Destructor
  ~DensityFit();

  /// Compute integrals
  void fill(const BasisSet & orbbas, const BasisSet & auxbas, bool direct);
  /// Compute index in integral table
  size_t idx(size_t ia, size_t imu, size_t inu) const;

  /// Compute estimate of necessary memory
  size_t memory_estimate(const BasisSet & orbbas, const BasisSet & auxbas, bool direct) const;

  /// Compute expansion coefficients c
  arma::vec compute_expansion(const arma::mat & P) const;

  /// Get Coulomb matrix from P
  arma::mat calc_J(const arma::mat & P) const;

  /// Get exchange matrix from orbitals with occupation numbers occs
  arma::mat calc_K(const arma::mat & C, const std::vector<double> & occs, size_t memlimit) const;
  
  /// Get the number of auxiliary functions
  size_t get_Naux() const;
  /// Get the three-electron integral
  double get_a_munu(size_t ia, size_t imu, size_t inu) const;
  /// Get ab_inv
  arma::mat get_ab_inv() const;
};


#endif
