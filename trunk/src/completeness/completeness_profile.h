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

#ifndef ERKALE_COMPPROF
#define ERKALE_COMPPROF

#include "../global.h"
#include "../basislibrary.h"

#include <armadillo>
#include <vector>

/// Structure for completeness profile
typedef struct {
  /// Angular momentum
  int am;
  /// Values of completeness profile \f$ Y(\alpha) \f$
  arma::vec Y;
} compprof_am_t;

/// Completeness profiles for a given element
typedef struct {
  /// Logarithms of scanning exponents \f$ \log_{10} \alpha \f$
  arma::vec lga;
  /// Completeness profiles for all angular momenta
  std::vector<compprof_am_t> shells;
} compprof_t;

/// Get scanning exponents
arma::vec get_scanning_exponents(double min, double max, size_t Np);

/// Compute overlap of normalized Gaussian primitives
arma::mat overlap(const arma::vec & z, const arma::vec & zp, int am);

/**
 * Compute completeness profile for element with given scanning exponents
 *
 * D. P. Chong, "Completeness profiles of one-electron basis sets",
 * Can. J. Chem. 73 (1995), pp. 79 - 83.
 */
compprof_t compute_completeness(const ElementBasisSet & bas, const arma::vec & scanexps, bool coulomb=false);

/**
 * Compute completeness profile for element from \f$ \alpha = 10^{min}
 * \f$ to \f$ \alpha = 10^{max} \f$ with \f$ N_p \f$ points
 *
 * D. P. Chong, "Completeness profiles of one-electron basis sets",
 * Can. J. Chem. 73 (1995), pp. 79 - 83.
 */
compprof_t compute_completeness(const ElementBasisSet & bas, double min=-10.0, double max=10.0, size_t Np=2001, bool coulomb=false);

#endif
