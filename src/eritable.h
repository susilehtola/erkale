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

#include "global.h"
#include "scf.h"

#ifndef ERKALE_ERITABLE
#define ERKALE_ERITABLE

// Debug usage of ERIs?
//#define CHECKFILL

#include <armadillo>
#include <cfloat>
#include <vector>
// Forward declaration
class BasisSet;

/// Helper for parallellizing loops
typedef struct {
  /// First basis function
  size_t i;
  /// Second basis function
  size_t j;
} bfpair_t;

/**
 * \class ERItable
 *
 * \brief Table of electron repulsion integrals
 *
 * This class is used to store electron repulsion integrals in memory
 * and to form the Coulomb and exchange matrices. There is no special
 * indexing, so also zeros are stored.
 *
 * \author Susi Lehtola
 * \date 2011/05/12 18:35
 */
class ERItable {
 protected:
  /// Table of integrals
  std::vector<double> ints;

  /// Index helper
  std::vector<size_t> iidx;
  /// List of pairs
  std::vector<bfpair_t> pairs;

  /// Range separation parameter
  double omega;
  /// Fraction of long-range (i.e. exact) exchange
  double alpha;
  /// Fraction of short-range exchange
  double beta;

  /// Calculate index in integral table
  virtual size_t idx(size_t i, size_t j, size_t k, size_t l) const;

 public:
  /// Constructor
  ERItable();
  /// Destructor
  ~ERItable();

  /// Set range separation
  void set_range_separation(double omega, double alpha, double beta);
  /// Get range separation
  void get_range_separation(double & omega, double & alpha, double & beta);

  /// Fill table
  void fill(const BasisSet * basis, double tol=DBL_EPSILON);

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
  /// Get size of ERI table
  size_t get_N() const;

  /// Form Coulomb matrix
  arma::mat calcJ(const arma::mat & R) const;
  /// Form exchange matrix
  arma::mat calcK(const arma::mat & R) const;
};

#include "basis.h"

#endif
