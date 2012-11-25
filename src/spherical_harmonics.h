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



#ifndef ERKALE_SPHHARM
#define ERKALE_SPHHARM

#include <complex>
#include <vector>

/// Calculate value of \f$ Y_{l}^{m} (\cos \theta, \phi) = (-1)^m \sqrt{ \frac {2l +1} {4 \pi} \frac {(l-m)!} {(l+m)!} } P_l^m (\cos \theta) e^{i m \phi} \f$
std::complex<double> spherical_harmonics(int l, int m, double cth, double phi);

/// Calculate expansion coefficients for cartesian terms from solid harmonics expansion
std::vector< std::complex<double> > cplx_Ylm_coeff(int l, int m);

#endif
