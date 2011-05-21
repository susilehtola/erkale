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



#include "global.h"

#ifndef ERKALE_OBARASAIKA
#define ERKALE_OBARASAIKA

#include <armadillo>

#include "basis.h"

/// Compute shell of overlap integrals
arma::mat overlap_int_os(double xa, double ya, double za, double zetaa, const std::vector<shellf_t> & carta, double xb, double yb, double zb, double zetab, const std::vector<shellf_t> & cartb);
/// Compute overlap of unnormalized primitives at r_A and r_B
double overlap_int_os(double xa, double ya, double za, double zetaa, int la, int ma, int na, double xb, double yb, double zb, double zetab, int lb, int mb, int nb);
/// Worker function
double overlap_int_1d(double xa, double xb, double zetaa, double zetab, int la, int lb);
/// Get array of overlap integrals
arma::mat overlap_ints_1d(double xa, double xb, double zetaa, double zetab, int la, int lb);

/// Compute shell of kinetic energy integrals
arma::mat kinetic_int_os(double xa, double ya, double za, double zetaa, const std::vector<shellf_t> & carta, double xb, double yb, double zb, double zetab, const std::vector<shellf_t> & cartb);
/// Compute kinetic energy integral of unnormalized primitives at r_A and r_B
double kinetic_int_os(double xa, double ya, double za, double zetaa, int la, int ma, int na, double xb, double yb, double zb, double zetab, int lb, int mb, int nb);
/// Worker function
double kinetic_int_1d(double xa, double xb, double zetaa, double zetab, int la, int lb);

/// Compute matrix element of derivative operator
double derivative_int_1d(double xa, double xb, double zetaa, double zetab, int la, int lb, int eval);
/// Get array of matrix elements of derivative operator
arma::mat derivative_ints_1d(double xa, double xb, double zetaa, double zetab, int la, int lb, int eval);

/// Compute shell of nuclear attraction integrals
arma::mat nuclear_int_os(double xa, double ya, double za, double zetaa, const std::vector<shellf_t> & carta, double xnuc, double ynuc, double znuc, double xb, double yb, double zb, double zetab, const std::vector<shellf_t> & cartb);
/// Compute nuclear attraction integral of unnormalized primitives at r_A and r_B and nucleus at r_C
double nuclear_int_os(double xa, double ya, double za, double zetaa, int la, int ma, int na, double xnuc, double ynuc, double znuc, double xb, double yb, double zb, double zetab, int lb, int mb, int nb);
/// Compute a shell of NAIs
arma::mat nuclear_ints_os(double xa, double ya, double za, double zetaa, int am_a, double xnuc, double ynuc, double znuc, double xb, double yb, double zb, double zetab, int am_b);
/// Worker routine, computes NAIs through recursions
void nuclear_recursion(arma::cube & ints, double PAx, double PAy, double PAz, double PBx, double PBy, double PBz, double PCx, double PCy, double PCz, double gamma, int iang, int jang);


#endif
