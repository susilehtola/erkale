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

#ifndef ERKALE_COMPPROF
#define ERKALE_COMPPROF

#include <vector>
#include "../basislibrary.h"

/// Structure for completeness profile
typedef struct {
  /// Angular momentum
  int am;
  /// Values of completeness profile \f$ Y(\alpha) \f$
  std::vector<double> Y;
} compprof_am_t;

/// Completeness profiles for a given element
typedef struct {
  /// Logarithms of scanning exponents \f$ \log_{10} \alpha \f$
  std::vector<double> lga;
  /// Completeness profiles for all angular momenta
  std::vector<compprof_am_t> shells;
} compprof_t;

/// Get scanning exponents
std::vector<double> get_scanning_exponents(double min, double max, size_t Np);

/**
 * Compute completeness profile for element with given scanning exponents
 *
 * D. P. Chong, "Completeness profiles of one-electron basis sets",
 * Can. J. Chem. 73 (1995), pp. 79 - 83.
 */
compprof_t compute_completeness(const ElementBasisSet & bas, const std::vector<double> & scanexps);

/**
 * Compute completeness profile for element from \f$ \alpha = 10^{min}
 * \f$ to \f$ \alpha = 10^{max} \f$ with \f$ N_p \f$ points
 *
 * D. P. Chong, "Completeness profiles of one-electron basis sets",
 * Can. J. Chem. 73 (1995), pp. 79 - 83.
 */
compprof_t compute_completeness(const ElementBasisSet & bas, double min=-10, double max=10, size_t Np=2000);

#endif
