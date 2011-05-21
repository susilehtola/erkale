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

#include "global.h"

#ifndef ERKALE_ERITABLE
#define ERKALE_ERITABLE

// Debug usage of ERIs?
//#define CHECKFILL

#include <armadillo>
#include <vector>
// Forward declaration
class BasisSet;

/**
 * \class ERItable
 *
 * \brief Table of electron repulsion integrals
 *
 * This class is used to store electron repulsion integrals in memory
 * and to form the Coulomb and exchange matrices. There is no special
 * indexing, so also zeros are stored.
 *
 * \author Jussi Lehtola
 * \date 2011/05/12 18:35
 */

class ERItable {
  /// Table of integrals
  std::vector<double> ints;

  /// Index helper
  std::vector<size_t> iidx;
  /// Calculate index in integral table
  size_t idx(size_t i, size_t j, size_t k, size_t l) const;
 public:
  /// Constructor
  ERItable();
  /// Destructor
  ~ERItable();

  /// Fill table
  void fill(const BasisSet * basis=NULL);

  /// Compute number of integrals
  size_t N_ints(const BasisSet * basis) const;
  /// Compute estimate of necessary memory
  size_t memory_estimate(const BasisSet * basis) const;

  /// Get ERI from table
  double getERI(size_t i, size_t j, size_t k, size_t l) const;

  /// Print ERI table
  void print() const;

  /// Count nonzero elements
  size_t count_nonzero() const;

  /// Get ERI table
  std::vector<double> & get();

  /// Form Coulomb matrix
  arma::mat calcJ(const arma::mat & R) const;
  /// Form exchange matrix
  arma::mat calcK(const arma::mat & R) const;
  /// Form Coulomb and exchange matrices at the same time
  void calcJK(const arma::mat & R, arma::mat & J, arma::mat & K) const;
  /// Form Coulomb and exchange matrices at the same time, unrestricted case
  void calcJK(const arma::mat & Ra, const arma::mat & Rb, arma::mat & J, arma::mat & Ka, arma::mat & Kb) const;
};

#include "basis.h"

#endif
