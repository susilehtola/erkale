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

#ifndef INTEGRALS_H
#define INTEGRALS_H

#include "global.h"
#include <armadillo>

/// Basis function
typedef struct {
  /// Primary quantum number
  int n;
  /// Slater exponent
  double zeta;
  /// Angular momentum
  int l;
  /// z component
  int m;
} bf_t;

/// Form overlap matrix
arma::mat overlap(const std::vector<bf_t> & basis);
/// Form kinetic energy matrix
arma::mat kinetic(const std::vector<bf_t> & basis);
/// Form nuclear attraction matrix
arma::mat nuclear(const std::vector<bf_t> & basis, int Z);
/// Form Coulomb matrix
arma::mat coulomb(const std::vector<bf_t> & basis, const arma::mat & P);
/// Form exchange matrix
arma::mat exchange(const std::vector<bf_t> & basis, const arma::mat & P);

/// Form density-fitted Coulomb matrix
arma::mat coulomb(const std::vector<bf_t> & basis, const std::vector<bf_t> & fitbas, const arma::mat & P);

/// Calculate normalization factor
double normalization(int n, double z);
/// Calculate overlap integral
double overlap(int na, int nb, double za, double zb, int la, int ma, int lb, int mb);
/// Wrapper for the above
inline double overlap(bf_t i, bf_t j) {
  return overlap(i.n,j.n,i.zeta,j.zeta,i.l,i.m,j.l,j.m);
}

/// Three-function overlap
double three_overlap(int na, int nc, int nd, int la, int ma, int lc, int mc, int ld, int md, double za, double zc, double zd);
/// Wrapper for the above
inline double three_overlap(bf_t i, bf_t k, bf_t l) {
  return three_overlap(i.n,k.n,l.n,i.l,i.m,k.l,k.m,l.l,l.m,i.zeta,k.zeta,l.zeta);
}


/// Calculate kinetic energy integral
double kinetic(int na, int nb, double za, double zb, int la, int ma, int lb, int mb);
/// Calculate nuclear attraction integral
double nuclear(int na, int nb, double za, double zb, int la, int ma, int lb, int mb);

/// Calculate unnormalized repulsion integral
double ERI_unnormalized(int na, int nb, int nc, int nd, double za, double zb, double zc, double zd, int la, int ma, int lb, int mb, int lc, int mc, int ld, int md);
/// Wrapper for the above
inline double ERI_unnormalized(bf_t i, bf_t j, bf_t k, bf_t l) {
  return ERI_unnormalized(i.n,j.n,k.n,l.n,i.zeta,j.zeta,k.zeta,l.zeta,i.l,i.m,j.l,j.m,k.l,k.m,l.l,l.m);
}
/// Calculate electron repulsion integral
double ERI(int na, int nb, int nc, int nd, double za, double zb, double zc, double zd, int la, int ma, int lb, int mb, int lc, int mc, int ld, int md);
/// Wrapper for the above
inline double ERI(bf_t i, bf_t j, bf_t k, bf_t l) {
  return ERI(i.n,j.n,k.n,l.n,i.zeta,j.zeta,k.zeta,l.zeta,i.l,i.m,j.l,j.m,k.l,k.m,l.l,l.m);
}

/// Calculate electron repulsion integral using Gaussian expansion
double gaussian_ERI(int la, int ma, int lb, int mb, int lc, int mc, int ld, int md, double za, double zb, double zc, double zd, int nfit);
/// Wrapper for the above
inline double gaussian_ERI(bf_t i, bf_t j, bf_t k, bf_t l) {
  return gaussian_ERI(i.l,i.m,j.l,j.m,k.l,k.m,l.l,l.m,i.zeta,j.zeta,k.zeta,l.zeta,6);
}

#endif
