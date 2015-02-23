/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

/**
 * \class ERIscreen
 *
 * \brief On-the-fly calculation of electron repulsion integrals
 *
 * This class performs on-the-fly formations of the Hartree-Fock
 * exchange and Coulomb matrices, that are needed calculations of
 * large systems for which the integrals don't fit into memory.  The
 * integrals are screened adaptively by the use of the Schwarz
 * inequality.
 *
 * \author Susi Lehtola
 * \date 2011/05/12 19:44
*/


#include "global.h"

#ifndef ERKALE_ERISCREEN
#define ERKALE_ERISCREEN

#include <armadillo>
#include <vector>
// Forward declaration
class BasisSet;
struct eripair_t;

/// Screening of electron repulsion integrals
class ERIscreen {
  /// Prescreening table of shell integrals
  arma::mat screen;
  /// Integral pairs sorted by value
  std::vector<eripair_t> shpairs;

  /// Pointer to the used basis set
  const BasisSet * basp;
  /// Index helper
  std::vector<size_t> iidx;

  /// Range separation parameter
  double omega;
  /// Fraction of long-range exchange
  double alpha;
  /// Fraction of short-range exchange
  double beta;

 public:
  /// Constructor
  ERIscreen();
  /// Destructor
  ~ERIscreen();

  /// Set range separation
  void set_range_separation(double omega, double alpha, double beta);
  /// Get range separation
  void get_range_separation(double & omega, double & alpha, double & beta);

  /// Form screening matrix, return amount of significant shell pairs
  size_t fill(const BasisSet * basis, double shtol, bool verbose=true);

  /// Calculate Coulomb matrix with tolerance tol for integrals
  arma::mat calcJ(const arma::mat & R, double tol) const;
  /// Calculate exchange matrix with tolerance tol for integrals
  arma::mat calcK(const arma::mat & R, double tol) const;
  /// Calculate  exchange matrices with tolerance tol for integrals
  void calcK(const arma::mat & Ra, const arma::mat & Rb, arma::mat & Ka, arma::mat & Kb, double tol) const;
  /// Calculate Coulomb and exchange matrices at the same time with tolerance tol for integrals
  void calcJK(const arma::mat & R, arma::mat & J, arma::mat & K, double tol) const;
  /// Calculate Coulomb and exchange matrices at the same time with tolerance tol for integrals, unrestricted calculation
  void calcJK(const arma::mat & Ra, const arma::mat & Rb, arma::mat & J, arma::mat & Ka, arma::mat & Kb, double tol) const;

  /* Force calculation */

  /// Calculate Coulomb force with tolerance tol for integrals
  arma::vec forceJ(const arma::mat & R, double tol) const;
  /// Calculate exchange force with tolerance tol for integrals
  arma::vec forceK(const arma::mat & R, double tol) const;
  /// Calculate Coulomb and exchange forces at the same time with tolerance tol for integrals, unrestricted calculation
  void forceK(const arma::mat & Ra, const arma::mat & Rb, arma::vec & fKa, arma::vec & fKb, double tol) const;
  /// Calculate Coulomb and exchange forces at the same time with tolerance tol for integrals
  void forceJK(const arma::mat & R, arma::vec & fJ, arma::vec & fK, double tol) const;
  /// Calculate Coulomb and exchange forces at the same time with tolerance tol for integrals, unrestricted calculation
  void forceJK(const arma::mat & Ra, const arma::mat & Rb, arma::vec & fJ, arma::vec & fKa, arma::vec & fKb, double tol) const;

  /* Versions for SIC routines */

  /// Calculate Coulomb matrix with tolerance tol for integrals
  std::vector<arma::mat> calcJ(const std::vector<arma::mat> & R, double tol) const;
  /// Calculate Coulomb and exchange matrices at the same time with tolerance tol for integrals
  void calcJK(const std::vector<arma::mat> & R, std::vector<arma::mat> & J, std::vector<arma::mat> & K, double tol) const;
  /// Calculate Coulomb and exchange matrices at the same time with tolerance tol for integrals, unrestricted calculation
  void calcJK(const std::vector<arma::mat> & Ra, const std::vector<arma::mat> & Rb, std::vector<arma::mat> & J, std::vector<arma::mat> & Ka, std::vector<arma::mat> & Kb, double tol) const;
};

#include "basis.h"

#endif
