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

#ifndef ERKALE_EMDCUBE
#define ERKALE_EMDCUBE

#include <vector>
#include "../basis.h"

/**
 * Compute the EMD in a cube, save output to emdcube.dat
 */
void emd_cube(const BasisSet & bas, const arma::mat & P, const std::vector<double> & px, const std::vector<double> & py, const std::vector<double> & pz);


#endif
