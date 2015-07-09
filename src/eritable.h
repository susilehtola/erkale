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
#include "basis.h"

#ifndef ERKALE_ERITABLE
#define ERKALE_ERITABLE

// Debug usage of ERIs?
//#define CHECKFILL

#include <armadillo>
#include <cfloat>
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
 * \author Susi Lehtola
 * \date 2011/05/12 18:35
 */
class ERItable {
 protected:
  /// Integral pairs sorted by value
  std::vector<eripair_t> shpairs;
  /// Screening matrix
  arma::mat screen;
  /// Table of integrals
  std::vector<double> ints;
  /// Offset lookup
  std::vector<size_t> shoff;
  
  /// Range separation parameter
  double omega;
  /// Fraction of long-range (i.e. exact) exchange
  double alpha;
  /// Fraction of short-range exchange
  double beta;

  /// Calculate offset in integrals table
  size_t offset(size_t ip, size_t jp) const;
  
 public:
  /// Constructor
  ERItable();
  /// Destructor
  ~ERItable();

  /// Set range separation
  void set_range_separation(double omega, double alpha, double beta);
  /// Get range separation
  void get_range_separation(double & omega, double & alpha, double & beta) const;

  /// Fill table, return amount of significant shell pairs
  size_t fill(const BasisSet * basis, double thr);

  /// Compute number of integrals
  size_t N_ints(const BasisSet * basis, double thr);

  /// Print ERI table
  void print() const;

  /// Get size of ERI table
  size_t get_N() const;

  /// Form Coulomb matrix
  arma::mat calcJ(const arma::mat & P) const;
  /// Form exchange matrix
  arma::mat calcK(const arma::mat & P) const;
  /// Form exchange matrix
  arma::cx_mat calcK(const arma::cx_mat & P) const;
};

#endif
