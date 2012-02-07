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


#ifndef ERKALE_GAUNT
#define ERKALE_GAUNT

#include "global.h"
#include <armadillo>

/**
 * Computes Gaunt coefficients \f$ G^{M m m'}_{L l l'} \f$ in the expansion
 * \f$ Y_l^m (\Omega) Y_{l'}^{m'} (\Omega) = \sum_{L,M} G^{M m m'}_{L l l'} Y_L^M (\Omega) \f$
 */
double gaunt_coefficient(int L, int M, int l, int m, int lp, int mp);

/// Table of Gaunt coefficients
class Gaunt {
  /// Table of coefficients
  arma::cube table;
 public:
  /// Dummy constructor
  Gaunt();
  /// Constructor
  Gaunt(int Lmax, int lmax, int lpmax);
  /// Destructor
  ~Gaunt();

  /// Get Gaunt coefficient
  double coeff(int L, int M, int l, int m, int lp, int mp) const;
};


#endif
