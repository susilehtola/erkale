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

#ifndef ERKALE_INTEGRALS
#define ERKALE_INTEGRALS

/// Calculate normalization coefficient of Gaussian primitive (THO 2.2)
double normconst(double zeta, int l, int m, int n);

/// Calculate expansion coefficient of x^j in (x+a)^l (x+b)^m
double fj(int j, int l, int m, double a, double b);

/// Compute center of r_A and r_B
double center_1d(double zetaa, double xa, double zetab, double xb);

/// Compute distance squared of r_A and r_
double distsq(double xa, double ya, double za, double xb, double yb, double zb);
/// Compute distance of r_A and r_
double dist(double xa, double ya, double za, double xb, double yb, double zb);

/// Compute overlap of unnormalized primitives at r_A and r_B
double overlap_int(double xa, double ya, double za, double zetaa, int la, int ma, int na, double xb, double yb, double zb, double zetab, int lb, int mb, int nb);

/// Calculate kinetic energy integral
double kinetic_int(double xa, double ya, double za, double zetaa, int la, int ma, int na, double xb, double yb, double zb, double zetab, int lb, int mb, int nb);

/// Calculate nuclear attraction integral
double nuclear_int(double xa, double ya, double za, double zetaa, int la, int ma, int na, double xnuc, double ynuc, double znuc, double xb, double yb, double zb, double zetab, int lb, int mb, int nb);

/// Calculate electron repulsion integral
double ERI_int(int la, int ma, int na, double Ax, double Ay, double Az, double zetaa, int lb, int mb, int nb, double Bx, double By, double Bz, double zetab, int lc, int mc, int nc, double Cx, double Cy, double Cz, double zetac, int ld, int md, int nd, double Dx, double Dy, double Dz, double zetad);
#endif
