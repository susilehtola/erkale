/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


#ifndef ERKALE_FOURIERPROD
#define ERKALE_FOURIERPROD

#include "bfprod.h"
#include <complex>
#include "../global.h"
#include <armadillo>

/// Structure for contraction in transform
typedef struct {
  /// px exponent
  int l;
  /// py exponent
  int m;
  /// pz exponent
  int n;

  /// Contraction coefficient
  std::complex<double> c;
} prod_fourier_contr_t;

/// Operator for sorting
bool operator<(const prod_fourier_contr_t & lhs, const prod_fourier_contr_t & rhs);
/// Operator for addition
bool operator==(const prod_fourier_contr_t & lhs, const prod_fourier_contr_t & rhs);

/// Structure for Fourier transform of product of basis functions
typedef struct {
  /// x coordinate of center (phase factor)
  double xp;
  /// y coordinate of center (phase factor)
  double yp;
  /// z coordinate of center (phase factor)
  double zp;

  /// Exponent
  double zeta;

  /// Contraction
  std::vector<prod_fourier_contr_t> c;
} prod_fourier_t;

/// Operator for sorting
bool operator<(const prod_fourier_t & lhs, const prod_fourier_t & rhs);
/// Operator for addition
bool operator==(const prod_fourier_t & lhs, const prod_fourier_t & rhs);

class prod_fourier {
  /// Product Gaussians
  std::vector<prod_fourier_t> p;

  void add_term(const prod_fourier_t & t);
  void add_contr(size_t ind, const prod_fourier_contr_t & t);

 public:
  prod_fourier();
  prod_fourier(const prod_gaussian_3d & p);
  ~prod_fourier();

  /// Get complex conjugate
  prod_fourier conjugate() const;

  /// Multiplication operator
  prod_fourier operator*(const prod_fourier & rhs) const;
  /// Scaling operator
  prod_fourier operator*(double fac) const;
  /// Addition operator
  prod_fourier & operator+=(const prod_fourier & rhs);

  /// Get the expansion
  std::vector<prod_fourier_t> get() const;

  /// Evaluate at p=(px,py,pz)
  std::complex<double> eval(double px, double py, double pz) const;

  /// Print out expansion
  void print() const;
};

/// Fourier transform products
std::vector<prod_fourier> fourier_transform(const std::vector<prod_gaussian_3d> & prod);

/// Get momentum transfer matrix \f$ \langle \mu|\exp i {\bf q} \cdot {\bf r} |\nu \rangle \f$ using Fourier method
arma::cx_mat momentum_transfer(const std::vector<prod_fourier> & fprod, size_t Nbf, const arma::vec & q);

#endif
