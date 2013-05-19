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



#include "global.h"

#ifndef ERKALE_SOLIDHARMONICS
#define ERKALE_SOLIDHARMONICS

#include <armadillo>
#include <vector>

/// Get transformation matrix (Y_lm, cart)
arma::mat Ylm_transmat(int l);

/**
 * Computes cartesian coefficients of Y_lm.
 *
 * Based on the article
 * http://en.wikipedia.org/wiki/Solid_spherical_harmonics
 */
std::vector<double> calcYlm_coeff(int l, int m);

#endif
