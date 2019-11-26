/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2019
 * Copyright (c) 2010-2019, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_PIV_CHOL_BASIS
#define ERKALE_PIV_CHOL_BASIS

BasisSetLibrary pivoted_cholesky_basis(const std::vector<atom_t> & atoms, const BasisSetLibrary & orig, double thresh);

#endif
