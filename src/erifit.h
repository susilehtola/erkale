/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Copyright © 2015 The Regents of the University of California 
 * All Rights Reserved 
 *
 * Written by Susi Lehtola, Lawrence Berkeley National Laboratory
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_ERIFIT
#define ERKALE_ERIFIT

#include "global.h"
#include "basislibrary.h"
#include <set>

namespace ERIfit {
  /// Basis function pair
  struct bf_pair_t {
    /// Index
    size_t idx;
    /// lh function
    size_t i;
    /// shell index
    size_t is;
    /// rh function
    size_t j;
    /// shell index
    size_t js;
  };

  /// Comparison operator
  bool operator<(const bf_pair_t & lhs, const bf_pair_t & rhs);

  /// Compute the exact repulsion integrals
  void compute_ERIs(const BasisSet & basis, arma::mat & eris);
  /// Compute the exact repulsion integrals
  void compute_ERIs(const ElementBasisSet & orbel, arma::mat & eris);
  /// Compute the exact diagonal repulsion integrals
  void compute_diag_ERIs(const ElementBasisSet & orbel, arma::mat & eris);

  /// Find unique exponent pairs
  void unique_exponent_pairs(const ElementBasisSet & orbel, int am1, int am2, std::vector< std::vector<shellpair_t> > & pairs, std::vector<double> & exps);
  /// Compute the T matrix needed for Cholesky decomposition
  void compute_cholesky_T(const ElementBasisSet & orbel, int am1, int am2, arma::mat & eris, arma::vec & exps);

  /// Compute fitting integrals
  void compute_fitint(const BasisSetLibrary & fitlib, const ElementBasisSet & orbel, arma::mat & fitint);
    
  /// Compute the fitted repulsion integrals using the supplied fitting integrals
  void compute_ERIfit(const BasisSetLibrary & fitlib, const ElementBasisSet & orbel, double linthr, const arma::mat & fitint, arma::mat & fiteri);
  /// Compute the diagonal fitted repulsion integrals using the supplied fitting integrals
  void compute_diag_ERIfit(const BasisSetLibrary & fitlib, const ElementBasisSet & orbel, double linthr, const arma::mat & fitint, arma::mat & fiteri);

  /// Compute the transformation matrix to orthonormal orbitals
  void orthonormal_ERI_trans(const ElementBasisSet & orbel, double linthr, arma::mat & trans);
}

#endif
