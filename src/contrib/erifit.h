/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2015
 * Copyright (c) 2010-2015, Susi Lehtola
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

  /// Compute the exact integrals
  void compute_ERIs(const ElementBasisSet & orbel, arma::mat & eris);

  /// Compute the fitted integrals. 
  void compute_ERIfit(const BasisSetLibrary & fitlib, const ElementBasisSet & orbel, double linthr, arma::mat & fitint, arma::mat & fiteri);
}

#endif
