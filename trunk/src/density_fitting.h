/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


/**
 * \class DensityFit
 *
 * \brief Density fitting routines
 *
 * This class contains density fitting routines used for the
 * approximate calculation of the Coulomb operator J. It is normally
 * used in pure DFT calculations, in which Hartree-Fock exchange does
 * not need to be calculated.
 *
 * The implementation is based on the procedure described in
 *
 * K. Eichkorn, O. Treutler, H. Öhm, M. Häser and R. Alrichs,
 *  "Auxiliary basis sets to approximate Coulomb potentials",
 * Chem. Phys. Lett. 240 (1995), 283-290.
 * 
 * Integral screening has not yet been implemented.
 *
 * \author Jussi Lehtola
 * \date 2011/04/12 17:52
 */




#ifndef ERKALE_DENSITYFIT
#define ERKALE_DENSITYFIT

#include "global.h"
#include "basis.h"

/// Density fitting routines
class DensityFit {
  /// Amount of orbital basis functions
  size_t Norb;
  /// Amount of auxiliary basis functions
  size_t Naux;
  /// Direct calculation? (Compute three-center integrals on-the-fly)
  bool direct;

  /// Orbital shells
  std::vector<GaussianShell> orbshells;
  /// Density fitting shells
  std::vector<GaussianShell> auxshells;
  
  /// Index helper
  std::vector<size_t> iidx;

  /// Screening matrix
  arma::mat screen;

  /// Integrals \f$ ( \alpha | \mu \nu) \f$
  std::vector<double> a_munu;
  /// \f$ (\alpha|\beta) \f$
  arma::mat ab;
  /// \f$ ( \alpha | \beta)^-1 \f$
  arma::mat ab_inv;

  /// Gamma
  arma::vec gamma;

  /// Expansion coefficients
  arma::vec c;

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

  /// Update density
  void update_density(const arma::mat & P);

  /// Get Coulomb matrix
  arma::mat calc_J() const;
};


#endif
