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



#ifndef ERKALE_CHEBYSHEV
#define ERKALE_CHEBYSHEV

#include <vector>
#include <cstdlib>

/// Modified Gauss-Chebyshev quadrature of the second kind for calculating \f$\int_{-1}^{1} f(x) dx\f$
void chebyshev(int n, std::vector<double> & x, std::vector<double> & w);

/// Modified Gauss-Chebyshev quadrature of the second kind for calculating \f$\int_{0}^{\infty} f(r) dr\f$. NB! For integration in spherical coordinates, you need to plug in the r^2 factor as well.
void radial_chebyshev(int n, std::vector<double> & r, std::vector<double> & wr);

/// Modified Gauss-Chebyshev quadrature of the second kind for calculating \f$\int_{0}^{\infty} r^2 f(r) dr\f$, i.e., this includes the r^2 factor.
void radial_chebyshev_jac(int n, std::vector<double> & r, std::vector<double> & wr);

#endif
