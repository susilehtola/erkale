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

// Modified Gauss-Chebyshev quadrature of the second kind for calculating \f$\int_{-1}^{1} f(x) dx\f$
void chebyshev(int n, std::vector<double> & x, std::vector<double> & w);

#endif
