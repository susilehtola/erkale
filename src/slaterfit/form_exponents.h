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

#ifndef FORM_EXPONENTS_H
#define FORM_EXPONENTS_H

#include <vector>
#include "basis.h"

/**
 * Fit Slater function with Gaussians.
 *
 * method is 0 for even-tempered, 1 for well-tempered and 2 for full optimization
 */
std::vector<contr_t> slater_fit(double zeta, int am, int nf, bool verbose=true, int method=2);

#endif
