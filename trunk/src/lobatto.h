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



#ifndef ERKALE_LOBATTO
#define ERKALE_LOBATTO

#include<vector>

/// Compute a Gauss-Lobatto quadrature rule for \f$ \int_{-1}^1 f(x)dx \approx \frac 2 {n(n-1)} \left[ f(-1) + f(1) \right] + \sum_{i=2}^{n-1} w_i f(x_i) \f$
void lobatto_compute ( int n, std::vector<double> & x, std::vector<double> & w);

#endif
