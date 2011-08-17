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



#ifndef ERKALE_SPHHARM
#define ERKALE_SPHHARM

#include <complex>

/// Calculate value of \f$ Y_{l}^{m} (\cos \theta, \phi) \f$
std::complex<double> spherical_harmonics(int l, int m, double cth, double phi);

#endif
