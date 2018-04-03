/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ATOMTABLE_H
#define ATOMTABLE_H

#include "integrals.h"

/// Helper for parallellizing loops
typedef struct {
  /// First basis function
  size_t i;
  /// Second basis function
  size_t j;
} bfpair_t;

class AtomTable {
  /// Amount of functions
  size_t Nbf;
  /// Calculate index in integral table
  size_t idx(size_t i, size_t j, size_t k, size_t l) const;
  /// List of pairs
  std::vector<bfpair_t> pairs;
  /// Table of integrals
  std::vector<double> ints;

 public:
  /// Consructor
  AtomTable();
  /// Destructor
  ~AtomTable();

  /// Fill table
  void fill(const std::vector<bf_t> & bas, bool verbose);
  /// Get ERI from table
  double getERI(size_t i, size_t j, size_t k, size_t l) const;

  /// Form Coulomb matrix
  arma::mat calcJ(const arma::mat & P) const;
  /// Form exchange matrix
  arma::mat calcK(const arma::mat & P) const;
};

#endif
