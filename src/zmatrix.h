/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2015
 * Copyright (c) 2010-2015, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


#ifndef ERKALE_ZMAT
#define ERKALE_ZMAT

#include "global.h"
#include "xyzutils.h"
#include <string>

/// Load z matrix from file, converting angstrom to au
std::vector<atom_t> load_zmat(std::string filename, bool convert);

#endif
