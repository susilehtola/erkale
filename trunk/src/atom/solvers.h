/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef SOLVERS_H
#define SOLVERS_H

#include "integrals.h"
#include "scf.h"

void RHF(const std::vector<bf_t> & basis, int Z, rscf_t & sol, const convergence_t conv, bool verbose=true);
void UHF(const std::vector<bf_t> & basis, int Z, uscf_t & sol, const convergence_t conv, bool ROHF=false, bool verbose=true);

#endif
