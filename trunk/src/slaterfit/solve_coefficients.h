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

#ifndef ERKALE_SLATER_COEFF
#define ERKALE_SLATER_COEFF

#include <armadillo>
#include <vector>

/**
 * Solve optimal coefficients to fit STO with exponent zeta with given
 * Gaussian exponents
 *
 * K. O-ohata, H. Taketa and S. Huzinaga, "Gaussian Expansions of
 * Atomic Orbitals", J. Phys. Soc. Japan 21 (1966), pp. 2036 - 2313.
 */
arma::vec solve_coefficients(std::vector<double> expns, double zeta, int l);

/// Compute self-overlap of difference
double compute_difference(std::vector<double> expns, double zeta, int l);

#endif
