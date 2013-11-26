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



#ifndef ERKALE_TEMPERED
#define ERKALE_TEMPERED

#include "global.h"
#include <cstddef>
#include <vector>
#include <armadillo>

/// Form a set of even-tempered exponents: \f$ \zeta_i = \alpha \beta^{i-1}, \ i=1,\dots,N_f \f$
std::vector<double> eventempered_set(double alpha, double beta, int Nf);

/**
 * Form a well-tempered set of exponents with
 * \f$\zeta_1 = \alpha \f$
 * \f$\zeta_2 = \alpha*\beta \f$
 * \f$\zeta_n = \zeta_{n-1} \beta \left[ 1 + \gamma \left( \frac n N \right)^\delta \right] \f$
 *
 * for a reference, see e.g.
 *
 * S. Huzinaga and M. Klobukowski, "Well-tempered Gaussian basis sets
 * for the calculation of matrix Hartree-Fock wavefunctions",
 * Chem. Phys. Lett. 212 (1993), pp. 260 - 264.
 */
std::vector<double> welltempered_set(double alpha, double beta, double gamma, double delta, size_t Nf);

/**
 * Expansion in orthonormal Legendre polynomials:
 * \f$ \ln \alpha_j = \sum_k P_k \left( \frac {2j-2} {N_p -1} - 1 \right) \f$
 *
 * G. Petersson, S. Zhong, J. A. Montgomery and M. J. Frish, "On the
 * optimization of Gaussian basis sets", J. Chem. Phys. 118, 1101
 * (2003).
 *
 * Input: $A$ values, and amount of functions
 * Output: actual values of exponents $\alpha_j$.
 */
arma::vec legendre_set(const arma::vec & A, int Nf);

#endif
