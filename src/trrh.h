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

#ifndef ERKALE_TRRH
#define ERKALE_TRRH

#include "global.h"
#include <armadillo>

/**
 * Trust-region Roothaan-Hall minimizer.
 *
 * L. Thøgersen et al., "The trust-region self-consistent field method
 * in Kohn–Sham density-functional theory", J. Chem. Phys. 123, 074103
 * (2005).
 *
 * Input: Fock matrix F, MO coefficients C, overlap matrix S, number of occupied states nocc
 * Output: new orbital coefficients Cnew and pseudo-orbital energies Enew
 */
void TRRH_update(const arma::mat & F_AO, const arma::mat & C, const arma::mat & S, arma::mat & Cnew, arma::vec & Enew, size_t nocc, bool verbose, double minovl);
/// Same, but for complex matrices
void TRRH_update(const arma::cx_mat & F_AO, const arma::cx_mat & C, const arma::mat & S, arma::cx_mat & Cnew, arma::vec & Enew, size_t nocc, bool verbose, double minovl);

#endif
