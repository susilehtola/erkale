/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


#ifndef ERKALE_MOMSER
#define ERKALE_MOMSER

#include "basis.h"
#include "global.h"
#include <armadillo>
#include <vector>

/// Class for computing momentum transfer matrices using a series expansion
class momentum_transfer_series {
  /// Stack of moment matrices: stack[l][ind](i,j)
  std::vector< std::vector< arma::mat > > stack;

  /// Maximum l in stack
  int lmax;

  /// Pointer to basis set
  const BasisSet * bas;
 public:
  /// Constructor
  momentum_transfer_series(const BasisSet * bas);
  /// Destructor
  ~momentum_transfer_series();

  /// Evaluate matrix for given value of q, within RMS tolerance rmstol and maximum tolerance maxtol
  arma::cx_mat get(const arma::vec & q, double rmstol, double maxtol);
};


#endif
